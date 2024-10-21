document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector('form');
    const emailInput = document.querySelector('input[name="email_text"]');

    form.addEventListener("submit", function(event) {
        event.preventDefault(); // Prevent the default form submission

        const emailText = emailInput.value;

        // Fetch predictions from the Flask backend
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ email_text: emailText })
        })
        .then(response => response.json())
        .then(data => {
            // Display predictions on the page
            let predictionHtml = '';
            if (data.prediction_nb) {
                predictionHtml += `<h2>Naive Bayes Prediction: ${data.prediction_nb}</h2>`;
            }
            if (data.prediction_dt) {
                predictionHtml += `<h2>Decision Tree Prediction: ${data.prediction_dt}</h2>`;
            }
            document.querySelector('.container').innerHTML += predictionHtml;
        })
        .catch(error => console.error('Error:', error));
    });
});
