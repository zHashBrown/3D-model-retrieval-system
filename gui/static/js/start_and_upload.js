function start() {
	let load_ranges = $("#modelclass").val();
	let filepath = $('#location').val();
	if (filepath == '') {
		alert('请先选择要检索的文件');
		return;
	}
	if (load_ranges == null) {				//这里是null而不是''，和添加单选的''不一样
		alert('注意，您没有选择检索范围！');
		return;
	}
	$('#startbtn').html('&nbsp;模型检索中...');
	$.get("/retrieval_data/" + filepath + '/' + load_ranges, {}, function(ret) {
		deal_and_show(ret);
		$('#startbtn').html('&nbsp;开始检索');
	})
}

function uptocloud() {
	let filename = $('#location').val();
	if (filename == '') {
		alert('请先选择要上传的文件');
		return;
	}
	let fileFormat = filename.substring(filename.lastIndexOf(".")).toLowerCase();
	// 检查是否是图片
	if (fileFormat.match(/.png|.jpg|.jpeg|.bmp/)) {
		alert('上传失败！文件必须为三维模型（支持格式：obj、off、stl、3ds、fbx、ply）');
		return;
	} else if (fileFormat.match(/.off|.obj|.stl|.3ds|.fbx|.x3d|.ply/)) {			//改名仅上传obj文件
		filename = filename.split(".")[0]+'.obj';
	}
	$('#tocloudbtn').html('&nbsp;模型上传中...');
	$.get("/uptocloud/" + filename, {}, function(ret) {
		alert(ret);
		$('#tocloudbtn').html('&nbsp;上传到数据库');
	})
}

function displayall(load_ranges) {
	$(".savebtn").css('display', 'none');
	$.get("/displayall/" + load_ranges, {}, function(ret) {
		deal_and_show(ret)
	})
}

function deal_and_show(ret){

	let retrival_list;
	let model_all_name;
	let model_class;
	show_list = [];
	all_pages=0;
	num_last_p = 0;//最后一页显示的模型数

	if (ret.length > 0) {

		$("#pages").css('display','block');
		$(".detail").css('display','block');
		retrival_list = ret.split(',');
		model_all_name = '';
		model_class = '';

		model_all_name = retrival_list[0].substring(1, retrival_list[0].length - 1);
		model_class = model_all_name.split('-')[0];			//分类名里不要有-减号字符
		show_list[0] = "static/models/obj/ModelNet40/" + model_class + "/" + model_all_name;
		for (let i = 1; i < retrival_list.length; i++) {
			model_all_name = retrival_list[i].substring(2, retrival_list[i].length - 1); //由于python的list输出转str，这里带一个空格，从2开始
			model_class = model_all_name.split('-')[0];
			show_list[i] = "static/models/obj/ModelNet40/" + model_class + "/" + model_all_name;
		}
		all_pages = Math.ceil(show_list.length / n);
		num_last_p = show_list.length % n;
		if(num_last_p==0){
			num_last_p = n;
		}
		draw(show_list, 1)
	} else {
		alert("返回值为空，可能查询失败，请重试！");
		window.location.reload();
	}
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