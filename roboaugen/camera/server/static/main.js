'use strict';

const setup_controller_connection = function() {
    const socket = io.connect( {transports: ['websocket']});
    socket.on('camera_updated', function(data) {
        const visual_predictions_initial = document.getElementById("visual_predictions_initial")
        visual_predictions_initial.src= 'data:image/png;base64,' + data.visual_predictions_initial;

        const visual_predictions_final = document.getElementById("visual_predictions_final")
        visual_predictions_final.src= 'data:image/png;base64,' + data.visual_predictions_final;

        const image_initial_procrustes = document.getElementById("image_initial_procrustes")
        image_initial_procrustes.src= 'data:image/png;base64,' + data.image_initial_procrustes;

        const image_final_procrustes = document.getElementById("image_final_procrustes")
        image_final_procrustes.src= 'data:image/png;base64,' + data.image_final_procrustes;

        const image_initial_grouped = document.getElementById("image_initial_grouped")
        image_initial_grouped.src= 'data:image/png;base64,' + data.image_initial_grouped;

        const image_final_grouped = document.getElementById("image_final_grouped")
        image_final_grouped.src= 'data:image/png;base64,' + data.image_final_grouped;

    });
};

$(document).ready(function(){
    setup_controller_connection();
});