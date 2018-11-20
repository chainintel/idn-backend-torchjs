'use strict';
var expect = require('chai').expect;
const { TorchJSBackend } = require('../dist/index.js');
const { fromBuffer, toBuffer } = require('@idn/util-buffer');

describe('TorchJSBackend', () => {
  it('should run', async () => {
    var backend = new TorchJSBackend(['torchjs/cuda']);
    let runner = await backend.init({
      modelType: 'torchjs/cuda',
      modelPath: 'http://cloudflare-ipfs.com/ipfs/QmfWy5PnoeVdEdMzn6ZZMwrG9NzmDKsTiAbMs5qSufGGgM',
      inputs: [{ shape: [1, 1, 28, 28] }]
    });
    let result = await runner.infer([toBuffer(new Float32Array(784))]);
    console.log(result);
    // expect(result).to.equal('Boys');
  });
});
