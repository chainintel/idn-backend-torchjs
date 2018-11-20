import { Backend } from '@idn/backend';

const { fromBuffer, toBuffer } = require('@idn/util-buffer');
const torchjs = require('@idn/torchjs');
const download = require('download');
const path = require('path');

/**
 * Currently only support single input / output
 */
class TorchJSBackend extends Backend {
  constructor(supported_types) {
    super(supported_types);
  }
  async _initFn(model) {
    let modelPath = model.path;
    // run the first type that support
    let type;
    let i = 0;
    while (i < model.types.length) {
      if (this.types.indexOf(model.types[i]) >= 0) {
        type = model.types[i];
        break;
      }
      i = i + 1;
    }
    if (!type) {
      throw 'NO SUPPORTED TYPES AVAILABLE';
    }

    let [_, backend] = type.split('/');
    await download(modelPath, './_model');
    var runner = new torchjs.ScriptModule(path.join('./_model', path.basename(modelPath)));
    if (backend === 'cuda') {
      runner.cuda();
      // trigger cuda complilation
      if (model.inputs) {
        let data = model.inputs.map((inp) => {
          return torchjs.ones(inp.shape, false);
        });
        runner.forward(data[0]);
      }
    }
    runner.model = model;
    return runner;
  }
  async _inferFn(runner, inputs) {
    let inp = runner.model.inputs[0];
    let data = torchjs.zeros(inp.shape, false);
    data.data = inputs[0];
    let output = await runner.forward(data);
    let outputs = [toBuffer(output.data)];
    return outputs;
  }
  async _destroyFn(runner, model) {}
}

export { TorchJSBackend };
