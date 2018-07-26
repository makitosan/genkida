console.log('loaded genkida.js');

import Vue from 'vue';
import Editor from './components/Editor.vue';
import Example from './components/Example.vue';

import Jimp from 'jimp/browser/lib/jimp';
// import Buffer from 'buffer';
import * as Posenet from '@tensorflow-models/posenet';
import * as TF from '@tensorflow/tfjs';

Vue.component('my-editor', Editor);
Vue.component('my-example', Example);

Vue.prototype.$posenet = Posenet;
Vue.prototype.$tf = TF;

new Vue({
  el: '#app',
  data: function() {
    return {
      msg: 'genkida.js',
    };
  },
});
