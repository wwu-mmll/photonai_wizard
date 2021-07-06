function color_background(){
    $(".line-highlight" ).css({'background': '#d0cffa', 'opacity': 0.7}).delay(800).fadeOut(1000);
}

function fetchdata(){
    console.log("ajax");
    $.ajax({
            url: '/update_code/' + stepName,
            type: 'post',
            data: $('#photon_step_form').serialize(),
            success: function (response) {

                $('#photon_code').html(response["photon_code"])
                $('#photon_code_input').attr("value", response["photon_code"]);
                $('#python-syntax-pre').attr('data-line', response["line_numbers"]);
                Prism.highlightElement($('#photon_code')[0]);
                color_background();
                setTimeout(color_background, 500);

            }
        }
    )

}

var timer = null;
var timeoutDuration = 1*1000;
function setPostTimeout(){
    if(setTimer==true){
        clearTimeout(timer);
        timer = setTimeout(fetchdata, timeoutDuration);
    }
}
$( document ).ready(function() {

    var validator = $('#photon_step_form').validate();

    $( ".photon-radio-label" ).click(function() {
        console.log("click");
        setPostTimeout();
    });

    $(".uk-input").change(function(){
        console.log("text changed");
        setPostTimeout();
    })

});
