async function init() {
  const video = document.getElementById('video');
  const result = document.getElementById('result');
  const confidenceFill = document.getElementById('confidenceFill');

  // Start camera
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { result.textContent = "Camera error: " + err; });

  // Load AI model
  const model = await tf.loadLayersModel('model.json');
  result.textContent = "Model loaded. Detecting...";

  // Labels for different plastic types + quality
  const labels = [
    'PP_Good','PP_Bad','PE_Good','PE_Bad',
    'PVC_Good','PVC_Bad','Mixed1_Good','Mixed1_Bad',
    'Mixed2_Good','Mixed2_Bad'
  ];

  // Keep predicting every 500ms
  setInterval(async () => {
    const img = tf.browser.fromPixels(video)
      .resizeNearestNeighbor([224,224])
      .toFloat()
      .div(255.0)
      .expandDims();
    const prediction = await model.predict(img).data();
    img.dispose();

    let maxIndex = prediction.indexOf(Math.max(...prediction));
    let conf = prediction[maxIndex] * 100;

    result.textContent = `${labels[maxIndex]} (${conf.toFixed(1)}%)`;

    confidenceFill.style.width = `${conf}%`;
    confidenceFill.style.backgroundColor =
      conf > 80 ? 'green' : conf > 50 ? 'orange' : 'red';
  }, 500);
}

init();
