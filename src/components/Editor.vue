<template>
  <div class="container">
    <div id="single" style="display:none"><canvas ></canvas></div>
    <div id="result"><img id="result_img"/></div>
    <div>
      <button @click="genki">GENKI</button>
    </div>
    <div>
      x: <input v-model.number="overlayProperties.x" type="number">
      y: <input v-model.number="overlayProperties.y" type="number">
    </div>
    <input accept="image/*" id="input_img" type="file" @change="fileChanged">
  </div>
</template>

<script>
  export default {
    name: 'Editor',
    data: function () {
      return {
        srcUrl: null,
        originalImage: null, // not-resized original image
        originalOverlayImage: null, // no-resized overlay image
        overlayProperties: {
          x: 0,
          y: 0,
          w: 0,
          h: 0
        },
        srcImage: null,
        imgNode: null,
        // TS related values
        net: null,
        posenetImage: null,
        modelOutputs: null,
        POINT_COLOR: 'red',
        LINE_WIDTH: 2
      }
    },
    mounted () {
      this.loadPosenet();
    },
    methods: {
      fileChanged: function(event) {
        if (!event.target.files.length) return;
        let file = event.target.files[0]
        let filereader = new FileReader();
        filereader.onload = function() {
          this.imgLoad(filereader.result, this.net);

          this.genki();
        }.bind(this);
        filereader.readAsDataURL(file);
      },
      loadImage: function () {
        Jimp.read(this.srcUrl).then(function (lenna) {
          this.originalImage = lenna.clone()
          this.srcImage = lenna
          lenna.resize(350, 234)
            .getBase64(Jimp.MIME_PNG, function (err, src) {
              this.imgNode.setAttribute('src', src)

              err && console.log(err)
            }.bind(this))
        }.bind(this)).catch(function (err) {
          console.error(err)
        })
      },
      genki: function () {
        let url = document.getElementById('parts_genkidama')
        this.overlay(url.src, 0.8)
        // this.overlayProperties.w = 50
        // this.overlayProperties.h = 50
      },
      overlay: function(url, opacity = 1) {
        Jimp.read(url).then(function (overlay) {
          this.originalOverlayImage = overlay.clone()
          this.srcImage = this.originalImage.clone()
          overlay.resize(this.overlayProperties.w, this.overlayProperties.h)
          overlay.opacity(opacity)

          this.srcImage.composite(overlay,
            this.overlayProperties.x - (this.overlayProperties.w / 2),
            this.overlayProperties.y - (this.overlayProperties.h / 2))
            .getBase64(Jimp.MIME_PNG, function (err, src) {
              this.imgNode.setAttribute('src', src)

              err && console.log(err)
            }.bind(this))
        }.bind(this)).catch(function (err) {
          console.error(err)
        })

      },
      upload: function() {
        let original = this.originalImage.clone()
        let overlay = this.originalOverlayImage.clone()
        original.composite(overlay, 0, 0)
          .getBase64(Jimp.MIME_PNG, function (err, src) {
            let params = new URLSearchParams()
            params.append('img', src)
            this.$http.post('/api/file/save', params)
              .then(res =>  {
                console.log('upload completed')
              })
              .catch(error => {
                console.log('error')
              });
          }.bind(this))
      },
      loadPosenet: async function() {
        console.log('loading posenet start');
        this.net = await this.$posenet.load();
        console.log('loading posenet end');
      },
      imgLoad: async function(src, net) {
        // Purge prevoius variables and free up GPU memory
        this.disposeModelOutputs();

        // Load an example image
        this.posenetImage = await this.loadImageFromLocal(src);

        // Creates a tensor from an image
        const input = this.$tf.fromPixels(this.posenetImage);

        // Stores the raw model outputs from both single- and multi-pose results can
        // be decoded.
        // Normally you would call estimateSinglePose or estimateMultiplePoses,
        // but by calling this method we can previous the outputs of the model and
        // visualize them.
        this.modelOutputs = await this.net.predictForMultiPose(input, 16);

        // Process the model outputs to convert into poses
        await this.decodeSinglePoseAndDrawResults();

        // setStatusText('');
        // document.getElementById('results').style.display = 'block';
        input.dispose();
      },
      /**
       * Purges variables and frees up GPU memory using dispose() method
       */
      disposeModelOutputs: function() {
        if (this.modelOutputs) {
          this.modelOutputs.heatmapScores.dispose();
          this.modelOutputs.offsets.dispose();
          this.modelOutputs.displacementFwd.dispose();
          this.modelOutputs.displacementBwd.dispose();
        }
      },
      loadImageFromLocal: async function(src) {
        const image = new Image();
        const promise = new Promise((resolve, reject) => {
          image.crossOrigin = '';
          image.onload = () => {
            resolve(image);
          };
        });
        image.src = src;
        return promise;
      },
      decodeSinglePoseAndDrawResults: async function() {
        if (!this.modelOutputs) {
          return;
        }

        const pose = await this.$posenet.decodeSinglePose(
          this.modelOutputs.heatmapScores, this.modelOutputs.offsets, 16);
        this.drawSinglePoseResults(pose);
      },
      /**
       * Draw the results from the single-pose estimation on to a canvas
       */
      drawSinglePoseResults: function(pose) {
        const canvas = document.querySelector('#single canvas');
        this.drawResults(canvas, [pose], 0.2, 0.2);

        // const {part, showHeatmap, showOffsets} = guiState.visualizeOutputs;
        const part = 0;
        const showHeatmap = 0;
        const showoffsets = 0;
        // displacements not used for single pose decoding
        const showDisplacements = false;
        const partId = +part;

        // visualizeOutputs(
        //   partId, showHeatmap, showOffsets, showDisplacements,
        //   canvas.getContext('2d'));
      },
      drawResults: function(canvas, poses, minPartConfidence, minPoseConfidence) {
        this.renderImageToCanvas(this.posenetImage, [this.posenetImage.width, this.posenetImage.height], canvas);
        poses.forEach((pose) => {
          if (pose.score >= minPoseConfidence) {
            this.drawKeypoints(pose.keypoints, minPartConfidence, canvas.getContext('2d'));
            this.drawSkeleton(pose.keypoints, minPartConfidence, canvas.getContext('2d'));

            // size is 3,4 ears distance
            this.overlayProperties.w = Math.abs(pose.keypoints[3].position.x - pose.keypoints[4].position.x);
            this.overlayProperties.h = this.overlayProperties.w;

            // 9 or 10
            // console.log('x' + pose.keypoints[10].position.x);
            if(pose.keypoints[10].position.y < pose.keypoints[9].position.y) {
              this.overlayProperties.x = pose.keypoints[10].position.x;
              this.overlayProperties.y = pose.keypoints[10].position.y - this.overlayProperties.w;
            } else {
              this.overlayProperties.x = pose.keypoints[9].position.x;
              this.overlayProperties.y = pose.keypoints[9].position.y - this.overlayProperties.w;
            }
          }
        });

        let url = canvas.toDataURL().replace(/^data:image\/\w+;base64,/, "");
        let buffer = new Buffer(url, 'base64');

        // https://github.com/oliver-moran/jimp/issues/231
        Jimp.read(buffer.buffer).then(function (lenna) {
          console.log('w:' + lenna.bitmap.width + ' h:' + lenna.bitmap.height);
          this.originalImage = lenna.clone();
          this.srcImage = lenna;
          lenna
            .getBase64(Jimp.MIME_PNG, function (err, src) {
              this.imgNode = document.getElementById('result_img');
              this.imgNode.setAttribute('src', src)

              err && console.log(err)
            }.bind(this))
        }.bind(this)).catch(function (err) {
          console.error(err)
        });
      },
      // ------------------------------------------------------------
      // utility functions
      // ------------------------------------------------------------
      renderImageToCanvas: function(image, size, canvas) {
        canvas.width = size[0];
        canvas.height = size[1];
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
      },

      /**
       * Draws a pose skeleton by looking up all adjacent keypoints/joints
       */
      drawSkeleton: function(keypoints, minConfidence, ctx, scale = 1) {
        const adjacentKeyPoints = this.$posenet.getAdjacentKeyPoints(keypoints, minConfidence);
        adjacentKeyPoints.forEach((keypoints) => {
          this.drawSegment(this.toTuple(keypoints[0].position),
            this.toTuple(keypoints[1].position), this.POINT_COLOR, scale, ctx);
        });
      },
      /**
       * Draw pose keypoints onto a canvas
       */
      drawKeypoints: function(keypoints, minConfidence, ctx, scale = 1) {
        for (let i = 0; i < keypoints.length; i++) {
          const keypoint = keypoints[i];

          if (keypoint.score < minConfidence) {
            continue;
          }
          const {y, x} = keypoint.position;
          this.drawPoint(ctx, y * scale, x * scale, 3, this.POINT_COLOR);
        }
      },
      toTuple: function({y, x}) {
        return [y, x];
      },
      drawPoint: function(ctx, y, x, r, color) {
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
      },
      /**
       * Draws a line on a canvas, i.e. a joint
       */
      drawSegment: function([ay, ax], [by, bx], color, scale, ctx) {
        ctx.beginPath();
        ctx.moveTo(ax * scale, ay * scale);
        ctx.lineTo(bx * scale, by * scale);
        ctx.lineWidth = this.LINE_WIDTH;
        ctx.strokeStyle = color;
        ctx.stroke();
      }
    }
  }
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  h1, h2 {
    font-weight: normal;
  }
  ul {
    list-style-type: none;
    padding: 0;
  }
  li {
    display: inline-block;
    margin: 0 10px;
  }
  a {
    color: #42b983;
  }
</style>
