// Fetch historical data and render a line chart using Chart.js
document.addEventListener('DOMContentLoaded', function () {
    const nodeId = document.getElementById('historicalChart').getAttribute('data-node-id'); // Assuming nodeId is passed in the HTML

    // API endpoint to fetch historical data
    const url = `/nodes/historical_data/${nodeId}`;

    // Fetch data from the backend
    fetch(url)
        .then(response => response.json())
        .then(data => {
            // Prepare data for the chart
            const labels = Object.keys(data.temperature); // Assuming 'temperature' is a key in the data
            const temperatureValues = Object.values(data.temperature);
            const humidityValues = Object.values(data.humidity);
            const soilMoistureValues = Object.values(data.soil_moisture);

            // Create the chart
            const ctx = document.getElementById('historicalChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels, // X-axis labels (dates)
                    datasets: [
                        {
                            label: 'Temperature (°C)',
                            data: temperatureValues,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            fill: false,
                        },
                        {
                            label: 'Humidity (%)',
                            data: humidityValues,
                            borderColor: 'rgba(54, 162, 235, 1)',
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            fill: false,
                        },
                        {
                            label: 'Soil Moisture (%)',
                            data: soilMoistureValues,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            fill: false,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: 'Historical Data Trends'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Date'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Value'
                            }
                        }
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error fetching historical data:', error);
            alert('Unable to load historical data. Please try again later.');
        });
});
