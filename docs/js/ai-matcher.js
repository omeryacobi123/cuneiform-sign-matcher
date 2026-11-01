(function(global){
  "use strict";

  var MODEL_URL="models/cuneiform-encoder.onnx";
  var INDEX_URL="models/sign-index.json";
  var VECTOR_URL="models/sign-embeddings.f32";
  var session=null,index=null,vectors=null,loading=null;

  function load(){
    if(loading)return loading;
    loading=Promise.all([
      ort.InferenceSession.create(MODEL_URL,{executionProviders:["wasm"]}),
      fetch(INDEX_URL).then(function(r){if(!r.ok)throw new Error("AI index missing");return r.json();}),
      fetch(VECTOR_URL).then(function(r){if(!r.ok)throw new Error("AI vectors missing");return r.arrayBuffer();})
    ]).then(function(parts){
      session=parts[0];index=parts[1];vectors=new Float32Array(parts[2]);
      if(vectors.length!==index.count*index.dimensions)throw new Error("AI index size mismatch");
      return {count:index.count,dimensions:index.dimensions,architecture:index.architecture};
    });
    return loading;
  }

  function tensorFromCanvas(source){
    var size=224,c=document.createElement("canvas"),ctx=c.getContext("2d",{willReadFrequently:true});
    c.width=size;c.height=size;ctx.fillStyle="#fff";ctx.fillRect(0,0,size,size);ctx.drawImage(source,0,0,size,size);
    var pixels=ctx.getImageData(0,0,size,size).data;
    var out=new Float32Array(3*size*size),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225];
    for(var i=0;i<size*size;i++)for(var ch=0;ch<3;ch++)out[ch*size*size+i]=(pixels[i*4+ch]/255-mean[ch])/std[ch];
    return new ort.Tensor("float32",out,[1,3,size,size]);
  }

  function query(canvas,period,limit){
    return load().then(function(){
      return session.run({image:tensorFromCanvas(canvas)});
    }).then(function(outputs){
      var q=outputs.embedding.data,d=index.dimensions,scored=[];
      for(var i=0;i<index.entries.length;i++){
        var entry=index.entries[i];
        if(period!=="All periods"&&entry.period!==period)continue;
        var dot=0,off=i*d;
        for(var j=0;j<d;j++)dot+=q[j]*vectors[off+j];
        scored.push({cp:entry.cp,period:entry.period,score:dot});
      }
      scored.sort(function(a,b){return b.score-a.score;});
      var seen={},result=[];
      for(var k=0;k<scored.length&&result.length<limit;k++)if(!seen[scored[k].cp]){seen[scored[k].cp]=1;result.push(scored[k]);}
      return result;
    });
  }

  global.AIMatcher={load:load,query:query};
})(window);
