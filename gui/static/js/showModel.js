function MyinitRender() {
	Myrenderer = new THREE.WebGLRenderer({
		antialias: true
	});
	//告诉渲染器需要阴影效果
	Myrenderer.setClearColor(0x000000);
	document.getElementById('ShowModalLabelContext').appendChild(Myrenderer.domElement);
}

function MyinitCamera() {
	Mycamera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
	Mycamera.position.set(0, 40, 50);
	Mycamera.lookAt(new THREE.Vector3(0, 0, 0));
}

function MyinitScene() {
	Myscene = new THREE.Scene();
}

function MyinitGui() {
	//声明一个保存需求修改的相关数据的对象
	gui = {};
	var MydatGui = new dat.GUI();
	//将设置属性添加到gui当中，gui.add(对象，属性，最小值，最大值）
}

function MyinitLight() {
	Myscene.add(new THREE.AmbientLight(0x444444));
	Mylight = new THREE.PointLight(0xffffff);
	Mylight.position.set(0, 50, 0);
	//告诉平行光需要开启阴影投射
	Mylight.castShadow = true;
	Myscene.add(Mylight);
}

function MyinitModel(Mypath) {
	//辅助工具
	var Myhelper = new THREE.AxesHelper(50);
	Myscene.add(Myhelper);
	var MyobjLoader = new THREE.OBJLoader();
	//设置当前加载的纹理
	MyobjLoader.load(Mypath, function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		Myscene.add(object); //会把所有的模型都添加
	});
}


function MyinitControls() {

	Mycontrols = new THREE.OrbitControls(Mycamera, Myrenderer.domElement);
	// 如果使用animate方法时，将此函数删除
	//controls.addEventListener( 'change', render );
	// 使动画循环使用时阻尼或自转 意思是否有惯性
	Mycontrols.enableDamping = true;
	//动态阻尼系数 就是鼠标拖拽旋转灵敏度
	//controls.dampingFactor = 0.25;
	//是否可以缩放
	Mycontrols.enableZoom = true;
	//是否自动旋转
	Mycontrols.autoRotate = true;
	//设置相机距离原点的最远距离
	Mycontrols.minDistance = 1;
	//设置相机距离原点的最远距离
	Mycontrols.maxDistance = 200;
	//是否开启右键拖拽
	Mycontrols.enablePan = true;
}

function Myrender() {
	Myrenderer.render(Myscene, Mycamera);
	Myrenderer.setSize(870, 520);
}

//窗口变动触发的函数
function MyonWindowResize() {
	Mycamera.aspect = window.innerWidth / window.innerHeight;
	Mycamera.updateProjectionMatrix();
	Myrender();
}

function Myanimate() {
	//更新控制器
	Myrender();
	//更新性能插件
	Mycontrols.update();
	requestAnimationFrame(Myanimate);
}

function MyremoveCube(displaypath) {
	var MyallChildren = Myscene.children;
	var MylastObject = MyallChildren[MyallChildren.length - 1];
	Myscene.remove(MylastObject);
	MyinitModel(displaypath);
}

function MyshowModel(displaypath) {
	if(undefined == displaypath){
  		alert("请选择已加载的模型");
  		return;
	}
	$('#ShowModal').modal('show');
	$('#ShowModalLabel').text(displaypath.split('/').slice(-1));
	if (displayflag == false) {
		displayflag = true; //设置为场景中已经存在模型
		MyinitGui();
		MyinitRender();
		MyinitScene();
		MyinitCamera();
		MyinitLight();
		MyinitModel(displaypath);
		MyinitControls();
		Myanimate();
		window.onresize = MyonWindowResize;
	} else {
		MyremoveCube(displaypath);
	}
}

function download(displaypath) {
	if(undefined == displaypath){
  		alert("请选择已加载的模型");
  		return;
	}
	window.open('http://127.0.0.1:5000/'+displaypath);//部署到服务器之后可以改为域名
	//window.open(document.URL+displaypath);
}

function download_from_Modal() {
	let model_all_name = $('#ShowModalLabel').text();
	let model_class = model_all_name.split('-')[0];
	window.open('http://127.0.0.1:5000/static/models/obj/ModelNet40/'+
		model_class+'/'+model_all_name);//部署到服务器之后可以改为域名
}
