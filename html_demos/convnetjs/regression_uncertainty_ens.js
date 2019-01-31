var lix=2; // layer id of layer we'd like to draw outputs of
var prior_var_in // need to declare this globally
var noise_var_in 
var reg_type_in
var lambda_w1, lambda_b1, lambda_w2, lambda_b2 // l2 regularisation coefficients

function reload_reg() { // this fn restarts the NN

  var momentum_in = 0.2
  var l_rate_in = 0.5
  var opt_in = 'adagrad'

  // get which activation fn checked
  var act_in = ''
  var radios = document.getElementsByName('act_fn');
  for (var i = 0, length = radios.length; i < length; i++) {
      if (radios[i].checked) {
          act_in = radios[i].value;
          break;
      }
  }
  layer_defs[1].activation = act_in
  layer_defs_2[1].activation = act_in
  layer_defs_3[1].activation = act_in
  layer_defs_4[1].activation = act_in
  layer_defs_5[1].activation = act_in
  console.log('updated layer act fn: ' + layer_defs[1].activation)

  // get prior input
  var radios = document.getElementsByName('prior_var');
  for (var i = 0, length = radios.length; i < length; i++) {
      if (radios[i].checked) {
          prior_var_in = parseFloat(radios[i].value);
          break;
      }
  }
  console.log('updated prior_var_in: ' + prior_var_in)

  // get data noise var input
  var radios = document.getElementsByName('noise_var');
  for (var i = 0, length = radios.length; i < length; i++) {
      if (radios[i].checked) {
          noise_var_in = parseFloat(radios[i].value);
          break;
      }
  }
  console.log('updated noise_var_in: ' + noise_var_in)

  // get reg type input
  var radios = document.getElementsByName('reg_type');
  for (var i = 0, length = radios.length; i < length; i++) {
      if (radios[i].checked) {
          reg_type_in = radios[i].value;
          break;
      }
  }
  console.log('updated reg_type_in: ' + reg_type_in)

  // if(reg_type_in === 'anc'){
  //   l2_decay = noise_var_in / prior_var_in
  // } else if(reg_type_in === 'reg'){
  //   l2_decay = noise_var_in / prior_var_in
  // } else if(reg_type_in === 'uncons'){
  //   l2_decay = 0.0
  // }
  // console.log('updated l2_decay: ' + l2_decay)
  // // NO because this is not your prior var, need to vary by layer

  if(reg_type_in === 'anc' || reg_type_in === 'reg'){
    lambda_w1 = noise_var_in / (prior_var_in /n_hidden)
    lambda_b1 = noise_var_in / prior_var_in
    lambda_w2 = noise_var_in / (1.0 /n_hidden)
    lambda_b2 = noise_var_in / 0.01
  } else {
    lambda_w1 = 0.0
    lambda_b1 = 0.0
    lambda_w2 = 0.0
    lambda_b2 = 0.0
  }

  net = new convnetjs.Net();
  net.makeLayers(layer_defs); // this is defined in html file
  trainer = new convnetjs.SGDTrainer(net, {method:opt_in,learning_rate:l_rate_in, momentum:momentum_in, batch_size:N, l2_decay:l2_decay});

  net_2 = new convnetjs.Net();
  net_2.makeLayers(layer_defs_2);
  trainer_2 = new convnetjs.SGDTrainer(net_2, {method:opt_in, learning_rate:l_rate_in, momentum:momentum_in, batch_size:N, l2_decay:l2_decay});

  net_3 = new convnetjs.Net();
  net_3.makeLayers(layer_defs_3);
  trainer_3 = new convnetjs.SGDTrainer(net_3, {method:opt_in, learning_rate:l_rate_in, momentum:momentum_in, batch_size:N, l2_decay:l2_decay});

  net_4 = new convnetjs.Net();
  net_4.makeLayers(layer_defs_4);
  trainer_4 = new convnetjs.SGDTrainer(net_4, {method:opt_in, learning_rate:l_rate_in, momentum:momentum_in, batch_size:N, l2_decay:l2_decay});

  net_5 = new convnetjs.Net();
  net_5.makeLayers(layer_defs_5);
  trainer_5 = new convnetjs.SGDTrainer(net_5, {method:opt_in, learning_rate:l_rate_in, momentum:momentum_in, batch_size:N, l2_decay:l2_decay});


  sum_y = Array();
  sum_y_2 = Array();
  for(var x=0.0; x<=WIDTH; x+= density){
    sum_y.push(new cnnutil.Window(100, 0));
    sum_y_2.push(new cnnutil.Window(100, 0));
  }
  sum_y_sq = Array();
  sum_y_sq_2 = Array();
  for(var x=0.0; x<=WIDTH; x+= density){
    sum_y_sq.push(new cnnutil.Window(100, 0));
    sum_y_sq_2.push(new cnnutil.Window(100, 0));
  }
  acc = 0;


}
 
function regen_data() {

  sum_y = Array();
  sum_y_2 = Array();
  sum_y_3 = Array();
  sum_y_4 = Array();
  sum_y_5 = Array();
  for(var x=0.0; x<=WIDTH; x+= density){
    sum_y.push(new cnnutil.Window(100, 0));
    sum_y_2.push(new cnnutil.Window(100, 0));
    sum_y_3.push(new cnnutil.Window(100, 0));
    sum_y_4.push(new cnnutil.Window(100, 0));
    sum_y_5.push(new cnnutil.Window(100, 0));
  }

  sum_y_sq = Array();
  sum_y_sq_2 = Array();
  sum_y_sq_3 = Array();
  sum_y_sq_4 = Array();
  sum_y_sq_5 = Array();
  for(var x=0.0; x<=WIDTH; x+= density){
    sum_y_sq.push(new cnnutil.Window(100, 0));
    sum_y_sq_2.push(new cnnutil.Window(100, 0));
    sum_y_sq_3.push(new cnnutil.Window(100, 0));
    sum_y_sq_4.push(new cnnutil.Window(100, 0));
    sum_y_sq_5.push(new cnnutil.Window(100, 0));
  }
  acc = 0;

  // create actual data
  data = [];
  labels = [];
  for(var i=0;i<N;i++) {
    var x = Math.random()*1.; // Math.random() gives 0-1
    if (x > 0.6) {x += 0.2;}
    var w = randn(0, 0.03*0.03); // guess this is random normal
    var y = x + Math.sin(alpha*(x + w)) + Math.sin(beta*(x + w)) + w; 
    data.push([x]);
    labels.push([y]);
  }
  // console.log('data = ' + data);
  // console.log('type of data = ' + typeof data[0]); // object
  // // var data=[[0],[0.2],[0.4]]
  // // var labels=[[0],[0.1],[0.3]]
  // data[0] = [0.0]
  // data[1] = [0.1]
  // data[2] = [0.3]
  // data[3] = [0.4]
  // data[4] = [0.5]
  // it's been set up as object of objects not array

  // data = [[-0.3],[-0.1],[0.5],[0.55],[0.9]]
  // labels=[[-0.1*5],[-0.1*5],[0.2*5],[0.4*5],[0.6*5]]

  // console.log('after messing data = ' + data);
  // console.log('type of data = ' + typeof data[0]);
  // console.log('regen_data - sum_y = ' + sum_y);
  // console.log('regen_data - sum_y_2 = ' + sum_y_2);
  // console.log('regen_data - sum_y_sq = ' + sum_y_sq);
  // console.log('regen_data - sum_y__sq2 = ' + sum_y_sq_2);

  
}

function myinit(){

  // get options checked
  var act_in = ''
  var radios = document.getElementsByName('act_fn');
  for (var i = 0, length = radios.length; i < length; i++) {
      if (radios[i].checked) {
          act_in = radios[i].value;
          break;
      }
  }
  console.log('act_fn checked: '+act_in);

  // get options checked
  var reg_type = ''
  var radios = document.getElementsByName('reg_type');
  for (var i = 0, length = radios.length; i < length; i++) {
      if (radios[i].checked) {
          reg_type = radios[i].value;
          break;
      }
  }
  console.log('ens_type checked: '+reg_type);

  regen_data();
  reload_reg();
}
 
function update_reg(){
  // forward prop the data
  
  var netx = new convnetjs.Vol(1,1,1); // still can't really work out what Vol is?
  var netx_2 = new convnetjs.Vol(1,1,1); 
  var netx_3 = new convnetjs.Vol(1,1,1); 
  var netx_4 = new convnetjs.Vol(1,1,1); 
  var netx_5 = new convnetjs.Vol(1,1,1); 
  avloss = 0.0;

  for(var iters=0;iters<100;iters++) { // this many epochs //100 works well
    for(var ix=0;ix<N;ix++) { // for each data point

      // should randomise order really
      ix = randi(0,N) // sample with replacement

      netx.w = data[ix];
      netx_2.w = data[ix];
      netx_3.w = data[ix];
      netx_4.w = data[ix];
      netx_5.w = data[ix];

      // do training
      var stats = trainer.train(netx, labels[ix]); // I'm not printing the loss anymore
      var stats = trainer_2.train(netx_2, labels[ix]); // I'm not printing the loss anymore
      var stats = trainer_3.train(netx_3, labels[ix]); // I'm not printing the loss anymore
      var stats = trainer_4.train(netx_4, labels[ix]); // I'm not printing the loss anymore
      var stats = trainer_5.train(netx_5, labels[ix]); // I'm not printing the loss anymore
      avloss += stats.loss;
    }
  }
  avloss /= N*iters;

}

function draw_reg(){    
    ctx_reg.clearRect(0,0,WIDTH,HEIGHT);
    ctx_reg.fillStyle = 'rgb(220,220,220)';
    ctx_reg.fillRect(0,0,WIDTH,HEIGHT)
    // ctx_reg.fill();

    var netx = new convnetjs.Vol(1,1,1);
    var netx_2 = new convnetjs.Vol(1,1,1);
    var netx_3 = new convnetjs.Vol(1,1,1);
    var netx_4 = new convnetjs.Vol(1,1,1);
    var netx_5 = new convnetjs.Vol(1,1,1);

    // draw decisions in the grid
    // var draw_neuron_outputs = $("#layer_outs").is(':checked');
    
    // draw final decision
    // var neurons = [];
    // var neurons_2 = [];



    ctx_reg.globalAlpha = 0.5;
    ctx_reg.beginPath();
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) { // run a fwd pass across grid

      netx.w[0] = (x-WIDTH/2)/ssw;
      netx_2.w[0] = (x-WIDTH/2)/ssw;
      netx_3.w[0] = (x-WIDTH/2)/ssw;
      netx_4.w[0] = (x-WIDTH/2)/ssw;
      netx_5.w[0] = (x-WIDTH/2)/ssw;
      var a = net.forward(netx); 
      var a_2 = net_2.forward(netx_2); 
      var a_3 = net_3.forward(netx_3); 
      var a_4 = net_4.forward(netx_4); 
      var a_5 = net_5.forward(netx_5); 
      var y = a.w[0];
      var y_2 = a_2.w[0];
      var y_3 = a_3.w[0];
      var y_4 = a_4.w[0];
      var y_5 = a_5.w[0];


      sum_y[c] = y;
      sum_y_2[c] = y_2;
      sum_y_3[c] = y_3;
      sum_y_4[c] = y_4;
      sum_y_5[c] = y_5;

      sum_y_sq[c] = y*y;
      sum_y_sq_2[c] = y_2*y_2;
      sum_y_sq_3[c] = y_3*y_3;
      sum_y_sq_4[c] = y_4*y_4;
      sum_y_sq_5[c] = y_5*y_5;

      // sum_y[c].add(y);
      // sum_y_2[c].add(y_2);
      // sum_y_sq[c].add(y*y);
      // sum_y_sq_2[c].add(y_2*y_2);

      // if(draw_neuron_outputs) {
      //   neurons.push(net.layers[lix].out_act.w); // back these up
      //   neurons_2.push(net_2.layers[lix].out_act.w); // back these up
      // }

      // if(x===0) ctx_reg.moveTo(x, -y*ssh+HEIGHT/2);
      // else ctx_reg.lineTo(x, -y*ssh+HEIGHT/2);
      c += 1;
    }

    // data generating function
    // ctx_reg.stroke();
    // ctx_reg.strokeStyle = 'rgb(0,250,50)';
    // ctx_reg.globalAlpha = 0.75;
    // ctx_reg.beginPath();
    // for(var x=0.0; x<=WIDTH; x+= density) {
    //   var xdash = (x-WIDTH/2)/ssw;
    //   var y = xdash + Math.sin(alpha*(xdash)) + Math.sin(beta*(xdash)); 
    //   if(x===0) ctx_reg.moveTo(x, -y*ssh+HEIGHT/2);
    //   else ctx_reg.lineTo(x, -y*ssh+HEIGHT/2);
    // }

    acc += 1;

    // ctx_reg.stroke();
    // ctx_reg.globalAlpha = 1.;

    // // draw individual neurons on first layer
    // // I think this allows drawing of the dropout lines?
    // if(draw_neuron_outputs) {
    //   var NL = neurons.length;
    //   ctx_reg.strokeStyle = 'rgb(250,50,50)';
    //   for(var k=0;k<NL;k++) {
    //     ctx_reg.beginPath();
    //     var n = 0;
    //     for(var x=0.0; x<=WIDTH; x+= density) {
    //       if(x===0) ctx_reg.moveTo(x, -neurons[n][k]*ssh+HEIGHT/2);
    //       else ctx_reg.lineTo(x, -neurons[n][k]*ssh+HEIGHT/2);
    //       n++;
    //     }
    //     ctx_reg.stroke();
    //   }
    // }
  
    // draw axes
    ctx_reg.beginPath();
    ctx_reg.strokeStyle = 'rgb(20,20,20)';
    ctx_reg.lineWidth = 1;
    ctx_reg.globalAlpha = 0.3;
    ctx_reg.moveTo(0, HEIGHT/2);
    ctx_reg.lineTo(WIDTH, HEIGHT/2);
    ctx_reg.moveTo(WIDTH/2, 0);
    ctx_reg.lineTo(WIDTH/2, HEIGHT);
    ctx_reg.stroke();





    
    // Draw the uncertainty

    // basically trying to trace around the whole shape
    // start at left, trace along upper boundary, then back along bottom boundary

    for(var i = 1; i <= 3; i++) { // for each std dev
      ctx_reg.fillStyle = 'rgb(200,0,250)';
      ctx_reg.globalAlpha = 0.15; // control how many std devs, alpha of each
      ctx_reg.beginPath();
      var c = 0;
      var start = 0
      for(var x=0.0; x<=WIDTH; x+= density) {
        var mean = (sum_y[c] + sum_y_2[c] + sum_y_3[c] + sum_y_4[c] + sum_y_5[c])/5
        var mean_sq = (sum_y_sq[c] + sum_y_sq_2[c] + sum_y_sq_3[c] + sum_y_sq_4[c] + sum_y_sq_5[c])/5
        var variance = mean_sq - Math.pow(mean,2);
        // var std = Math.sqrt(variance);
        var var_total = variance + noise_var_in
        var std = Math.sqrt(var_total);

        // modify mean so upperbound
        mean += std * i;
        if(x===0) {start = -mean*ssh+HEIGHT/2; ctx_reg.moveTo(x, start); }
        else ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
        c += 1;
      }

      // console.log('variance: ' + typeof variance);
      // console.log('noise_var_in: ' + typeof noise_var_in);
      // console.log('var_total: ' + typeof var_total);

      var c = sum_y.length - 1;
      for(var x=WIDTH; x>=0.0; x-= density) {

        // var mean = (sum_y[c] + sum_y_2[c])/2
        // var mean_sq = (sum_y_sq[c] + sum_y_sq_2[c])/2
        var mean = (sum_y[c] + sum_y_2[c] + sum_y_3[c] + sum_y_4[c] + sum_y_5[c])/5
        var mean_sq = (sum_y_sq[c] + sum_y_sq_2[c] + sum_y_sq_3[c] + sum_y_sq_4[c] + sum_y_sq_5[c])/5
        var variance = mean_sq - Math.pow(mean,2);
        // var std = Math.sqrt(variance);
        var var_total = variance + noise_var_in
        var std = Math.sqrt(var_total);

        // modify mean so lower bound
        mean -= std * i;
        ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
        c -= 1;
      }
      ctx_reg.lineTo(0, start);
      ctx_reg.fill();
      // ctx_reg.stroke();

      // could outline edge of std dev
      // ctx_reg.strokeStyle = 'rgb(0,0,0)';
      // ctx_reg.globalAlpha = 1.;
      // ctx_reg.stroke();
    }
    // ctx_reg.strokeStyle = 'rgb(0,0,0)';
    // ctx_reg.globalAlpha = 1.;
    // ctx_reg.stroke();

        col_in = 'rgb(10,10,10)'
    alpha_in = 0.8

    // NN_1 mean
    ctx_reg.beginPath();
    ctx_reg.strokeStyle = col_in;
    ctx_reg.lineWidth = 1;
    ctx_reg.globalAlpha = alpha_in;
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y[c]; //.get_average();
      // console.log('NN 1 sum_y = ' + sum_y[c].get_average());
      if(x===0) ctx_reg.moveTo(x, -mean*ssh+HEIGHT/2);
      else ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();

    // NN_2 mean
    ctx_reg.beginPath();
    ctx_reg.globalAlpha = alpha_in;
    ctx_reg.strokeStyle = col_in;
    ctx_reg.lineWidth = 1;
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y_2[c]; //.get_average();
      // console.log('NN 2 sum_y_2 = ' + sum_y_2[c].get_average());
      if(x===0) ctx_reg.moveTo(x, -mean*ssh+HEIGHT/2);
      else ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();

    // NN_3 mean
    ctx_reg.beginPath();
    ctx_reg.globalAlpha = alpha_in;
    ctx_reg.strokeStyle = col_in;
    ctx_reg.lineWidth = 1;
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y_3[c]; //.get_average();
      // console.log('NN 2 sum_y_2 = ' + sum_y_2[c].get_average());
      if(x===0) ctx_reg.moveTo(x, -mean*ssh+HEIGHT/2);
      else ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();

    // NN_4 mean
    ctx_reg.beginPath();
    ctx_reg.globalAlpha = alpha_in;
    ctx_reg.strokeStyle = col_in;
    ctx_reg.lineWidth = 1;
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y_4[c]; //.get_average();
      // console.log('NN 2 sum_y_2 = ' + sum_y_2[c].get_average());
      if(x===0) ctx_reg.moveTo(x, -mean*ssh+HEIGHT/2);
      else ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();

    // NN_5 mean
    ctx_reg.beginPath();
    ctx_reg.globalAlpha = alpha_in;
    ctx_reg.strokeStyle = col_in;
    ctx_reg.lineWidth = 1;
    var c = 0;
    for(var x=0.0; x<=WIDTH; x+= density) {
      var mean = sum_y_5[c]; //.get_average();
      // console.log('NN 2 sum_y_2 = ' + sum_y_2[c].get_average());
      if(x===0) ctx_reg.moveTo(x, -mean*ssh+HEIGHT/2);
      else ctx_reg.lineTo(x, -mean*ssh+HEIGHT/2);
      c += 1;
    }
    ctx_reg.stroke();

    // draw datapoints
    // ctx_reg.beginPath();
    ctx_reg.strokeStyle = 'rgb(0,0,0)';
    // ctx_reg.fillStyle = 'rgb(140,0,250)';
    ctx_reg.fillStyle = 'rgb(255,255,255)'; // white
    ctx_reg.globalAlpha = 1.;
    ctx_reg.lineWidth = 3;
    for(var i=0;i<N;i++) {
      // ctx_reg.drawCircle(data[i]*ssw+WIDTH/2, -labels[i]*ssh+HEIGHT/2, 5.0);
      
      ctx_reg.beginPath();

      ctx_reg.arc(data[i]*ssw+WIDTH/2, -labels[i]*ssh+HEIGHT/2, 5.0, 0, Math.PI*2, true); 
      ctx_reg.closePath();
      ctx_reg.stroke();
      ctx_reg.fill();
    }    
    // ctx_reg.stroke();



    // ctx_reg.fillStyle = "blue";
    // ctx_reg.font = "bold 16px Arial";
    // // ctx_reg.fillText("average loss: " + avloss, 20, 20);
    // ctx_reg.fillText("variable here: " + avloss, 20, 20);
}

// function addPoint(x, y){
//   // add datapoint at location of click
//   alert($(NPGcanvas).width())
//   data.push([(x-$(NPGcanvas).width()/2)/ss]);
//   labels.push([-(y-$(NPGcanvas).height()/2)/ss]);
//   N += 1;
// }

function mouseClick(x, y, shiftPressed){  
  // add datapoint at location of click
  // alert(WIDTH);
  // alert($(NPGcanvas).width());
  // alert(ss);
  //alert(x);
  x = x / $(NPGcanvas).width() * WIDTH;
  y = y / $(NPGcanvas).height() * HEIGHT;
  data.push([(x-WIDTH/2)/ssw]);
  labels.push([-(y-HEIGHT/2)/ssh]);
  N += 1;
}