export default class
{
    constructor(root)
    {
        this.root = root;
        console.log("This is constructor.");
    }

    update(data){
        this.clear();
        //this.setHeader(headerColumns);
        this.setBody(data);
    }

    clear()
    {
        this.root.innerHTML = "";
    }

    // setHeader(headerColumns)
    // {
    //     this.root.insertAdjacentHTML("afterbegin", `
    //         <thead>
    //         <tr>
    //             ${headerColumns.map(text => `<th>${text}</th>`).join(" ") }
    //         </tr>
    //         </thead>
    //     `);
    // }
   

    setBody(result)
    {
        const x_label = [];
        const temperature = [];
        const humidity = [];
        const LWS = []
        const soil_temperature = []
        const soil_moisture = []
        const battery_status = []
        for(let i=1;i<result.data.length;i++)
        {
            x_label.push(result.data[i][8]);
            temperature.push(result.data[i][2]);
            humidity.push(result.data[i][3]);
            LWS.push(result.data[i][6]);
            soil_temperature.push(result.data[i][4]);
            soil_moisture.push(result.data[i][5]);
            battery_status.push(result.data[i][7]);
        }
        //console.log(x_label, temperature, humidity, LWS, soil_temperature, soil_moisture, battery_status);

        // const box = document.createElement("div");
        // box.id = "box";
        // box.className = "row";
        // this.root.appendChild(box);

        // const cbox = document.createElement("div");
        // cbox.id = "mycanvas1";
        // cbox.class = "myCanvas1";
        // cbox.setAttribute("style","height: 100%;width: 100%;background-color:white;");
        // box.appendChild(cbox);

        // var canvas1 = document.createElement('canvas');
        // canvas1.id = "mychart1";
        // canvas1.setAttribute("style","height: auto;width: auto;");
        // cbox.appendChild(canvas1);

        this.root.innerHTML = "<canvas style='height:auto; width:auto' id='myChart1'>";
        var cid = document.getElementById('myChart1');
        const ctx1 = document.getElementById('myChart1').getContext('2d');
        const graph_data1 = {
            type: 'line',
            data: {
                labels: x_label,
                datasets: [{
                    label: 'Temperature',
                    data: temperature,
                    backgroundColor: ['rgb(227, 252, 232)',],
                    borderColor: ['rgb(71, 191, 87)',],
                    borderWidth: 2
                }]
            },
            options: {
                title: {
                    text: "Temperature",
                    display: false,
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        };
        const myChart1 = new Chart(ctx1, graph_data1);
        const newGraphData1 = graph_data1.data.datasets[0].data;
        newGraphData1.shift();
        newGraphData1.push();

        cid.insertAdjacentHTML("afterend", `
        <canvas style='height:auto; width:auto' id='myChart2'>
        `);
        cid = document.getElementById('myChart2');
        const ctx2 = document.getElementById('myChart2').getContext('2d');
        const graph_data2 = {
            type: 'line',
            data: {
                labels: x_label,
                datasets: [{
                    label: 'Humidity',
                    data: humidity,
                    backgroundColor: ['rgb(227, 252, 232)',],
                    borderColor: ['rgb(71, 191, 87)',],
                    borderWidth: 2
                }]
            },
            options: {
                title: {
                    text: "Humidity",
                    display: false,
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        };
        const myChart2 = new Chart(ctx2, graph_data2);
        const newGraphData2 = graph_data2.data.datasets[0].data;
        newGraphData2.shift();
        newGraphData2.push();

        cid.insertAdjacentHTML("afterend", `
        <canvas style='height:auto; width:auto' id='myChart3'>
        `);
        cid = document.getElementById('myChart3');
        const ctx3 = document.getElementById('myChart3').getContext('2d');
        const graph_data3 = {
            type: 'line',
            data: {
                labels: x_label,
                datasets: [{
                    label: 'LWS',
                    data: LWS,
                    backgroundColor: ['rgb(227, 252, 232)',],
                    borderColor: ['rgb(71, 191, 87)',],
                    borderWidth: 2
                }]
            },
            options: {
                title: {
                    text: "LWS",
                    display: false,
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        };
        const myChart3 = new Chart(ctx3, graph_data3);
        const newGraphData3 = graph_data3.data.datasets[0].data;
        newGraphData3.shift();
        newGraphData3.push();



        cid.insertAdjacentHTML("afterend", `
        <canvas style='height:auto; width:auto' id='myChart4'>
        `);
        cid = document.getElementById('myChart4');
        const ctx4 = document.getElementById('myChart4').getContext('2d');
        const graph_data4 = {
            type: 'line',
            data: {
                labels: x_label,
                datasets: [{
                    label: 'Soil Temperature',
                    data: soil_temperature,
                    backgroundColor: ['rgb(227, 252, 232)',],
                    borderColor: ['rgb(71, 191, 87)',],
                    borderWidth: 2
                }]
            },
            options: {
                title: {
                    text: "Soil Temperature",
                    display: false,
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        };
        const myChart4 = new Chart(ctx4, graph_data4);
        const newGraphData4 = graph_data4.data.datasets[0].data;
        newGraphData4.shift();
        newGraphData4.push();

        cid.insertAdjacentHTML("afterend", `
        <canvas style='height:auto; width:auto' id='myChart5'>
        `);
        cid = document.getElementById('myChart5');
        const ctx5 = document.getElementById('myChart5').getContext('2d');
        const graph_data5 = {
            type: 'line',
            data: {
                labels: x_label,
                datasets: [{
                    label: 'Soil Moisture',
                    data: soil_moisture,
                    backgroundColor: ['rgb(227, 252, 232)',],
                    borderColor: ['rgb(71, 191, 87)',],
                    borderWidth: 2
                }]
            },
            options: {
                title: {
                    text: "Soil Moisture",
                    display: false,
                },
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        };
        const myChart5 = new Chart(ctx5, graph_data5);
        const newGraphData5 = graph_data5.data.datasets[0].data;
        newGraphData5.shift();
        newGraphData5.push();

        cid.insertAdjacentHTML("afterend", `
        <canvas style='height:auto; width:auto' id='myChart6'>
        `);
        const ctx6 = document.getElementById('myChart6').getContext('2d');
        const graph_data6 = {
            type: 'line',
            data: {
                labels: x_label,
                datasets: [{
                    label: 'Battery Status',
                    data: battery_status,
                    backgroundColor: ['rgb(227, 252, 232)',],
                    borderColor: ['rgb(71, 191, 87)',],
                    borderWidth: 2
                }]
            },
            options: {
                title: {
                    text: "Battery Status",
                    display: false,
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        };
        const myChart6 = new Chart(ctx6, graph_data6);
        const newGraphData6 = graph_data6.data.datasets[0].data;
        newGraphData6.shift();
        newGraphData6.push();
    }
}
//for table preview
// export default class
// {
//     constructor(root)
//     {
//         this.root = root;
//         console.log("This is constructor.");
//     }
//
//     update(data, headerColumns = []){
//         this.clear();
//         this.setHeader(headerColumns);
//         this.setBody(data);
//     }
//
//     clear()
//     {
//         this.root.innerHTML = "";
//     }
//
//     setHeader(headerColumns)
//     {
//         this.root.insertAdjacentHTML("afterbegin", `
//             <thead>
//             <tr>
//                 ${headerColumns.map(text => `<th>${text}</th>`).join(" ") }
//             </tr>
//             </thead>
//         `);
//     }
//
//     setBody(data)
//     {
//         const rowsHTML = data.map( row => {
//             return `
//             <tr>
//                 ${row.map(text =>  `<td>${text}</td>`).join(" ")}
//             </tr>`
//         });
//         // console.log(rowsHTML);
//         this.root.insertAdjacentHTML("beforeend",`
//             <tbody>
//                 ${rowsHTML.join("")}
//             </tbody>
//         `);
//     }
// }