--[[
  Copyright 2014 Google Inc. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
]]--

require "env"
include "data.lua"
include "utils/strategies.lua"
include "layers/MaskedLoss.lua"
include "layers/Embedding.lua"

local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn")
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')


function rhn(x, prev_c, prev_h, noise_i, noise_h)
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slice the n_gates dimension, i.e dimension 2
  local reshaped_noise_i = nn.Reshape(2,params.rnn_size)(noise_i)
  local reshaped_noise_h = nn.Reshape(2,params.rnn_size)(noise_h)
  local sliced_noise_i   = nn.SplitTable(2)(reshaped_noise_i)
  local sliced_noise_h   = nn.SplitTable(2)(reshaped_noise_h)
  -- Calculate all two gates
  local dropped_h_tab = {}
  local h2h_tab = {}
  local t_gate_tab = {}
  local c_gate_tab = {}
  local in_transform_tab = {}
  local s_tab = {}
  for layer_i = 1, params.recurrence_depth do
    local i2h        = {}
    h2h_tab[layer_i] = {}
    if layer_i == 1 then
      for i = 1, 2 do
        -- Use select table to fetch each gate
        local dropped_x         = local_Dropout(x, nn.SelectTable(i)(sliced_noise_i))
        dropped_h_tab[layer_i]  = local_Dropout(prev_h, nn.SelectTable(i)(sliced_noise_h))
        i2h[i]                  = nn.Linear(params.rnn_size, params.rnn_size)(dropped_x)
        h2h_tab[layer_i][i]     = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h_tab[layer_i])
      end
      t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(nn.CAddTable()({i2h[1], h2h_tab[layer_i][1]})))
      in_transform_tab[layer_i] = nn.Tanh()(nn.CAddTable()({i2h[2], h2h_tab[layer_i][2]}))
      c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i]))
      s_tab[layer_i]           = nn.CAddTable()({
        nn.CMulTable()({c_gate_tab[layer_i], prev_h}),
        nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
      })
    else
      for i = 1, 2 do
        -- Use select table to fetch each gate
        dropped_h_tab[layer_i]  = local_Dropout(s_tab[layer_i-1], nn.SelectTable(i)(sliced_noise_h))
        h2h_tab[layer_i][i]     = nn.Linear(params.rnn_size, params.rnn_size)(dropped_h_tab[layer_i])
      end
      t_gate_tab[layer_i]       = nn.Sigmoid()(nn.AddConstant(params.initial_bias, False)(h2h_tab[layer_i][1]))
      in_transform_tab[layer_i] = nn.Tanh()(h2h_tab[layer_i][2])
      c_gate_tab[layer_i]       = nn.AddConstant(1,false)(nn.MulConstant(-1, false)(t_gate_tab[layer_i]))
      s_tab[layer_i]           = nn.CAddTable()({
        nn.CMulTable()({c_gate_tab[layer_i], s_tab[layer_i-1]}),
        nn.CMulTable()({t_gate_tab[layer_i], in_transform_tab[layer_i]})
      })
    end
  end
  local next_h = s_tab[params.recurrence_depth]
  local next_c = prev_c
  return next_c, next_h
end

function lstm(i, prev_c, prev_h)
  function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function transfer_data(x)
  return x:cuda()
end

function local_Dropout(input, noise)
  return nn.CMulTable()({input, noise})
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local noise_x          = nn.Identity()()
  local noise_i          = nn.Identity()()
  local noise_h          = nn.Identity()()
  local noise_o          = nn.Identity()()
  local i                = {[0] = Embedding(symbolsManager.vocab_size,
                                            params.rnn_size)(x)}
  i[0] = local_Dropout(i[0], noise_x)
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  local noise_i_split    = {noise_i:split(params.layers)}
  local noise_h_split    = {noise_h:split(params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local n_i            = noise_i_split[layer_idx]
    local n_h            = noise_h_split[layer_idx]
    local next_c, next_h = rhn(i[layer_idx - 1], prev_c, prev_h, n_i, n_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, symbolsManager.vocab_size)
  local dropped          = local_Dropout(i[params.layers], noise_o)
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = MaskedLoss()({pred, y})
  local module           = nn.gModule({x, y, prev_s, noise_x, noise_i, noise_h, noise_o},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

--function create_network()
--  local x                = nn.Identity()()
--  local y                = nn.Identity()()
--  local prev_s           = nn.Identity()()
--  local i                = {[0] = Embedding(symbolsManager.vocab_size,
--                                            params.rnn_size)(x)}
--  local next_s           = {}
--  local splitted         = {prev_s:split(2 * params.layers)}
--  for layer_idx = 1, params.layers do
--    local prev_c         = splitted[2 * layer_idx - 1]
--    local prev_h         = splitted[2 * layer_idx]
--    local dropped        = nn.Dropout()(i[layer_idx - 1])
--    local next_c, next_h = lstm(dropped, prev_c, prev_h)
--    table.insert(next_s, next_c)
--    table.insert(next_s, next_h)
--    i[layer_idx] = next_h
--  end
--  local h2y              = nn.Linear(params.rnn_size, symbolsManager.vocab_size)
--  local pred             = nn.LogSoftMax()(h2y(i[params.layers]))
--  local err              = MaskedLoss()({pred, y})
--  local module           = nn.gModule({x, y, prev_s},
--                                      {err, nn.Identity()(next_s)})
--  module:getParameters():uniform(-params.init_weight, params.init_weight)
--  if params.gpuidx > 0 then
--    return module:cuda()
--  else
--    return module
--  end
--end

function setup()
  print("Creating an RHN network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end

  model.noise_i = {}
  model.noise_x = {}
  model.noise_xe = {}
  for j = 1, params.seq_length do
    model.noise_x[j] = transfer_data(torch.zeros(params.batch_size, 1))
    model.noise_xe[j] = torch.expand(model.noise_x[j], params.batch_size, params.rnn_size)
    model.noise_xe[j] = transfer_data(model.noise_xe[j])
  end
  model.noise_h = {}
  for d = 1, params.layers do
    model.noise_i[d] = transfer_data(torch.zeros(params.batch_size, 2 * params.rnn_size))
    model.noise_h[d] = transfer_data(torch.zeros(params.batch_size, 2 * params.rnn_size))
  end
  model.noise_o = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))

  model.pred = {}
  for j = 1, params.seq_length do
    model.pred[j] = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
  end
  local y                = nn.Identity()()
  local pred             = nn.Identity()()
  local err              = nn.ClassNLLCriterion()({pred, y})
  model.test             = transfer_data(nn.gModule({y, pred}, {err}))
end

-- convenience functions to handle noise
local function sample_noise(state)
  for i = 1, params.seq_length do
    model.noise_x[i]:bernoulli(1 - params.dropout_x)
    model.noise_x[i]:div(1 - params.dropout_x)
  end

  for b = 1, params.batch_size do
    for i = 1, params.seq_length do
      local x = state.data[state.pos + i - 1]
      for j = i+1, params.seq_length do
        if state.data[state.pos + j - 1] == x then
          model.noise_x[j][b] = model.noise_x[i][b]
          -- we only need to override the first time; afterwards subsequent are copied:
          break
        end
      end
    end
  end
  for d = 1, params.layers do
    model.noise_i[d]:bernoulli(1 - params.dropout_i)
    model.noise_i[d]:div(1 - params.dropout_i)
    model.noise_h[d]:bernoulli(1 - params.dropout_h)
    model.noise_h[d]:div(1 - params.dropout_h)
  end
  model.noise_o:bernoulli(1 - params.dropout_o)
  model.noise_o:div(1 - params.dropout_o)
end

local function reset_noise()
  for j = 1, params.seq_length do
    model.noise_x[j]:zero():add(1)
  end
  for d = 1, params.layers do
    model.noise_i[d]:zero():add(1)
    model.noise_h[d]:zero():add(1)
  end
  model.noise_o:zero():add(1)
end


--function setup()
--  print("Creating a RNN LSTM network.")
--  local core_network = create_network()
--  paramx, paramdx = core_network:getParameters()
--  model = {}
--  model.s = {}
--  model.ds = {}
--  model.start_s = {}
--  for j = 0, params.seq_length do
--    model.s[j] = {}
--    for d = 1, 2 * params.layers do
--      model.s[j][d] = torch.zeros(params.batch_size, params.rnn_size)
--      if params.gpuidx > 0 then
--        model.s[j][d] = model.s[j][d]:cuda()
--      end
--
--    end
--  end
--  for d = 1, 2 * params.layers do
--    model.start_s[d] = torch.zeros(params.batch_size, params.rnn_size)
--    model.ds[d] = torch.zeros(params.batch_size, params.rnn_size)
--    if params.gpuidx > 0 then
--      model.start_s[d] = model.start_s[d]:cuda()
--      model.ds[d] = model.ds[d]:cuda()
--    end
--  end
--  model.core_network = core_network
--  model.rnns = cloneManyTimes(core_network, params.seq_length)
--  model.norm_dw = 0
--  reset_ds()
--end

function reset_state(state)
  load_data(state)
  state.pos = 1
  state.acc = 0
  state.count = 0
  state.normal = 0
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state, paramx_)
  if paramx_ ~= paramx then paramx:copy(paramx_) end
  copy_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data.x:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    tmp, model.s[i] = unpack(model.rnns[i]:forward({state.data.x[state.pos],
                                                    state.data.y[state.pos + 1],
                                                    model.s[i - 1]}))
    if params.gpuidx > 0 then
      cutorch.synchronize()
    end
    state.pos = state.pos + 1
    state.count = state.count + tmp[2]
    state.normal = state.normal + tmp[3]
  end
  state.acc = state.count / state.normal
  copy_table(model.start_s, model.s[params.seq_length])
end

function bp(state)
  paramdx:zero()
  reset_ds()
  local tmp_val
  if params.gpuidx > 0 then tmp_val = torch.ones(1):cuda() else tmp_val = torch.ones(1) end
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local tmp = model.rnns[i]:backward({state.data.x[state.pos],
                                        state.data.y[state.pos + 1],
                                        model.s[i - 1]},
                                        { tmp_val, model.ds})[3]
    copy_table(model.ds, tmp)
    if params.gpuidx > 0 then
      cutorch.synchronize()
    end
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
end

function eval_training(paramx_)
  fp(state_train, paramx_)
  bp(state_train)
  return 0, paramdx
end

function run_test(state)
  reset_state(state)
  for i = 1, (state.data.x:size(1) - 1) / params.seq_length do
    fp(state, paramx)
  end
end

function show_predictions(state)
  reset_state(state)
  copy_table(model.s[0], model.start_s)
  local input = {[1] = ""}
  local prediction = {[1] = ""}
  local sample_idx = 1
  local batch_idx = random(params.batch_size)
  for i = 1, state.data.x:size(1) - 1 do
    local tmp = model.rnns[1]:forward({state.data.x[state.pos],
                                              state.data.y[state.pos + 1],
                                              model.s[0]})[2]
    if params.gpuidx > 0 then
      cutorch.synchronize()
    end
    copy_table(model.s[0], tmp)
    local current_x = state.data.x[state.pos][batch_idx]
    input[sample_idx] = input[sample_idx] ..
                        symbolsManager.idx2symbol[current_x]
    local y = state.data.y[state.pos + 1][batch_idx]
    if y ~= 0 then
      local fnodes = model.rnns[1].forwardnodes
      local pred_vector = fnodes[#fnodes].data.mapindex[1].input[1][batch_idx]
      prediction[sample_idx] = prediction[sample_idx] ..
                               symbolsManager.idx2symbol[argmax(pred_vector)]
    end
    state.pos = state.pos + 1
    local last_x = state.data.x[state.pos - 1][batch_idx]
    if state.pos > 1 and symbolsManager.idx2symbol[last_x] == "." then
      if sample_idx >= 3 then
        break
      end
      sample_idx = sample_idx + 1
      input[sample_idx] = ""
      prediction[sample_idx] = ""
    end
  end
  io.write(string.format("Some exemplary predictions for the %s dataset\n",
                          state.name))
  for i = 1, #input do
    input[i] = input[i]:gsub("#", "\n\t\t     ")
    input[i] = input[i]:gsub("@", "\n\tTarget:      ")
    io.write(string.format("\tInput:\t     %s", input[i]))
    io.write(string.format("\n\tPrediction:  %s\n", prediction[i]))
    io.write("\t-----------------------------\n")
  end
end

function main()

  local cmd = torch.CmdLine()
  cmd:option('-gpuidx', 1, 'Index of GPU on which job should be executed. 0 for CPU')
  cmd:option('-target_length', 6, 'Length of the target expression.')
  cmd:option('-target_nesting', 3, 'Nesting of the target expression.')
  -- Available strategies: baseline, naive, mix, blend.
  cmd:option('-strategy', 'blend', 'Scheduling strategy.')
  cmd:text()
  local opt = cmd:parse(arg)

  params = {batch_size=100,
            seq_length=50,
            layers=1, --2
            rnn_size=400,
            init_weight=0.08,
            learningRate=0.5,
            max_grad_norm=5,
            target_length=opt.target_length,
            target_nesting=opt.target_nesting,
            target_accuracy=0.95,
            dropout_x=0.25,
            dropout_i=0.75,
            dropout_h=0.25,
            dropout_o=0.75,
            current_length=1,
            current_nesting=1,
            recurrence_depth=10,
            initial_bias=-4,
            gpuidx = opt.gpuidx }


--  local params = {batch_size=20,
--                seq_length=35,
--                layers=1,
--                decay=1.02,
--                rnn_size=1000,
--                dropout_x=0.25,
--                dropout_i=0.75,
--                dropout_h=0.25,
--                dropout_o=0.75,
--                init_weight=0.04,
--                lr=0.2,
--                vocab_size=10000,
--                max_epoch=20,
--                max_max_epoch=1000,
--                max_grad_norm=10,
--                weight_decay=1e-7,
--                recurrence_depth=10,
--                initial_bias=-4}

  init_gpu(opt.gpuidx)
  state_train = {hardness=_G[opt.strategy],
    len=math.max(10001, params.seq_length + 1),
    seed=1,
    kind=0,
    batch_size=params.batch_size,
    name="Training" }
  state_val =   {hardness=current_hardness,
    len=math.max(501, params.seq_length + 1),
    seed=1,
    kind=1,
    batch_size=params.batch_size,
    name="Validation" }

  state_test =  {hardness=target_hardness,
    len=math.max(501, params.seq_length + 1),
    seed=1,
    kind=2,
    batch_size=params.batch_size,
    name="Test"}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_val, state_test }

  for _, state in pairs(states) do
    reset_state(state)
    assert(state.len % params.seq_length == 1)
  end
  setup()
  local step = 0
  local epoch = 0
  local train_accs = {}
  local total_cases = 0
  local start_time = torch.tic()
  print("Starting training.")
  while true do
    local epoch_size = floor(state_train.data.x:size(1) / params.seq_length)
    step = step + 1
    if step % epoch_size == 0 then
      state_train.seed = state_train.seed + 1
      load_data(state_train)
    end
    optim.adam(eval_training, paramx, {learningRate=params.learningRate}, {})
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = ceil(step / epoch_size)
    if step % ceil(epoch_size / 2) == 10 then
      cps = floor(total_cases / torch.toc(start_time))
      run_test(state_val)
      run_test(state_test)
      local accs = ""
      for _, state in pairs(states) do
        accs = string.format('%s, %s acc.=%.2f%%',
          accs, state.name, 100.0 * state.acc)
      end
      print('epoch=' .. epoch .. accs ..
        ', current length=' .. params.current_length ..
        ', current nesting=' .. params.current_nesting ..
        ', characters per sec.=' .. cps ..
        ', learning rate=' .. string.format("%.3f", params.learningRate))
      if (state_val.acc > params.target_accuracy) or
        (#train_accs >= 5 and
        train_accs[#train_accs - 4] > state_train.acc) then
        if not make_harder() then
          params.learningRate = params.learningRate * 0.8
        end
        if params.learningRate < 1e-3 then
          break
        end
        load_data(state_train)
        load_data(state_val)
        train_accs = {}
      end
      train_accs[#train_accs + 1] = state_train.acc
      total_cases = 0
      start_time = torch.tic()
      show_predictions(state_train)
      show_predictions(state_val)
      show_predictions(state_test)
    end
    if step % 33 == 0 then
      collectgarbage()
    end
  end
  print("Training is over.")
end

main()
