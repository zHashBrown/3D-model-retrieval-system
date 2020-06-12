function get_tmp_files(){
    $.get("/get_tmp_files", {}, function(ret) {

		let retrival_list;
		let model_all_name;
		let model_upload_time;
		show_list = [];
		all_pages=0;
		num_last_p = 0;//最后一页显示的模型数

		if (ret.length > 0) {

		$(".savebtn").css('display', 'inline-block');
		$("#pages").css('display','block');
		$(".detail").css('display','block');

			retrival_list = ret.split(',');
			model_all_name = '';
			model_upload_time = '';

			model_all_name = retrival_list[0].substring(1, retrival_list[0].length - 1);
			model_upload_time = model_all_name.split('-')[0];			//分类名里不要有-减号字符
			show_list[0] = "static/user_upload/" + model_all_name;
			for (let i = 1; i < retrival_list.length; i++) {
				model_all_name = retrival_list[i].substring(2, retrival_list[i].length - 1); //由于python的list输出转str，这里带一个空格，从2开始
				model_upload_time = model_all_name.split('-')[0];
				show_list[i] = "static/user_upload/" + model_all_name;
			}
			all_pages = Math.ceil(show_list.length / n);
			num_last_p = show_list.length % n;
			if (num_last_p == 0) {
				num_last_p = n;
			}
			draw(show_list, 1)
		} else {
			alert("返回值为空，可能查询失败，请重试！");
			window.location.reload();
		}

	})
}


function delete_model(){
	let delete_model_name = $('#ShowModalLabel').text();
	$.get("/delete_model/"+delete_model_name, {}, function(ret) {
		alert('删除成功');
		window.location.reload();
	})
}

function before_add_model(displaypath){
	if(undefined == displaypath){
  		alert("请选择已加载的模型");
  		return;
	}
	$('#AddModal').modal('show')
	$('#AddModalName').text(displaypath.split('/').slice(-1))
}

function add_model(){
	let add_model_name = $('#AddModalName').text();
	let add_model_class = $("#addmodelclass").val();
	if (add_model_name == '待添加模型') {
		alert('请先选择要添加的文件');
		return;
	}
	if (add_model_class == '') {				//这里是''而不是null，和检索多选的null不一样
		alert('注意，您没有选择添加到的类别！');
		return;
	}
	$('#addbtn').html('&nbsp;模型添加中……');
	$.get("/add_model/"+add_model_name+'/'+add_model_class, {}, function() {
		alert('添加成功');
		window.location.reload();
	})
}

function login_out(){
	$.get("/clear_session", function(ret) {
		alert('管理员登出');
		window.location.href="/"
	})
}

function start_train(){
	let num_class = $("#class_num_input").val();
	if(num_class==''){
		alert('请输入类别数！');
		return
	}
	if(Math.floor(num_class) != num_class){
		alert('请输入整数！');
		return
	}
	$('#start_train_btn').html('&nbsp;模型训练中……');
	$.get("/start_train/"+ num_class, function(ret) {
		alert('模型已达90%精度！将自动停止训练');
		$('#start_train_btn').html('&nbsp;开始训练');
	})
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