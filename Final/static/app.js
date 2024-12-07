function handleClick() {
    fetch('/click', { method: 'POST' })  // Send POST request to the server
        .then(response => {
            if (response.ok) {
                console.log('User clicked on the video feed.');
            }
        })
        .catch(error => console.error('Error:', error));
}

const eventSource = new EventSource('/show_time');
eventSource.onmessage = function(event) {
    document.getElementById('current-time').innerText = event.data;
};

const drivingStatus = new EventSource('/get_driving_situation');
drivingStatus.onmessage = function(event) {
    collisionWarner = document.getElementById('collision');
    console.log(collisionWarner)
    console.log(collisionWarner.textContent)
    console.log(collisionWarner.innerText)
//    console.log(classList)

    // Parse the incoming event data
    const data = event.data;
//    console.log(data);

    // Update the text content and styles
    // Update the text content
    collisionWarner.textContent = data;
    if (data == "Driving Safely"){
        collisionWarner.classList.remove('warning');
    }
    else {
        // Add a class to apply the warning styles
        collisionWarner.classList.add('warning');
    }

};

const signImgDiv = document.getElementById('sign-img');

const trafficSigns = new EventSource('/get_traffic_signs');
trafficSigns.onmessage = function(event) {
    const signs = JSON.parse(event.data);
    console.log(signs)
    signImgDiv.innerHTML = ''; // Clear previous images

    // Create image elements for each sign
    signs.forEach(sign => {
        const img = document.createElement('img');
        img.src = `static/images/signs/${sign}.png`; // Path to images
        img.alt = sign;
        signImgDiv.appendChild(img);
    });
};