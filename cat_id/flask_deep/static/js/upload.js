function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#blah').attr('src', e.target.result).height(300);
            $('.inp_block').hide();
            $('.b_block').css('display','block');
            $('.c_block').css('display', 'block');
            return function(a){
                e.src = a.target.result;
            }
        };

        reader.readAsDataURL(input.files[0]);
    }
}
/*
$(document).ready(function(){
    $('.b_block').hide();
    $('.c_block').hide();
  });
*/