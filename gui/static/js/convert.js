function check_before_upload_convert() {
    let filename = $('#up_file_btn_convert').val();
    const obj = filename.lastIndexOf("\\");
    filename = filename.substr(obj + 1);
    if (filename == undefined || filename == '') {
        alert('未上传新文件！');
        $('#location').val('');
        return false;
    }
    let fileFormat = filename.substring(filename.lastIndexOf(".")).toLowerCase();

    if (fileFormat.match(/.off|.obj|.stl|.3ds|.fbx|.x3d|.ply/)) {
        $('#location').val(filename);
        $('#i-check').val('重新选择模型');
        return true;
    } else {
        alert('上传错误,文件格式必须为：off/obj/stl/3ds/fbx/ply');
        return false;
    }
}

function check_after_upload_convert(filename) {      //提交后的回调函数
    $('#startbtn').html('&nbsp;开始转换');
    if (filename=='') {
        alert('转换失败，请重试！');
        return;
    } else {
        alert('转换成功！');
	    window.open('http://127.0.0.1:5000/'+ 'static/convert/'+filename);//部署到服务器之后可以改为域名
    }
}

function check_and_upload_convert() {
    target_format = $("#target_format").val();
    let filepath = $('#location').val();
	if (filepath == '') {
		alert('请先上传要转换的模型！');
		return;
	}
	if (target_format == '') {				//这里是null而不是''，和添加单选的''不一样
		alert('注意，您没有选择目标格式！');
		return;
	}
	$('#startbtn').html('&nbsp;模型转换中...');
    $('#convert_Form').ajaxSubmit({
            type: 'post',
            url: '/file_convert/' + target_format,
            success: check_after_upload_convert,      //提交后的回调函数
        }
    )
}

function login() {
	let username = $('#username').val();
	let password = $('#password').val();
	if(username==''){
		$('#result').html('请输入用户名！');
	}else if(password==''){
		$('#result').html('请输入密码！');
	}
	else{
		$.post("/login/" + username +'/'+password , {}, function(ret) {
			if (ret != "undefined" && ret=="success") {
				window.location.href="admin";			//直接return render_template()不渲染
			}
			else {
					$('#result').html(ret);
                }
		})
	}
}