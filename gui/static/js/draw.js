function initRender() {
	for (let i = 0; i < n; i++) {
		renderer[i] = new THREE.WebGLRenderer({
			antialias: true
		});
		//告诉渲染器需要阴影效果
		renderer[i].setClearColor(0x000000);
		document.getElementById('pos' + i).appendChild(renderer[i].domElement);
	}
}

function initCamera() {
	for (let i = 0; i < n; i++) {
		camera[i] = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 1000);
		camera[i].position.set(0, 40, 50);
		camera[i].lookAt(new THREE.Vector3(0, 0, 0));
	}
}

function initScene() {
	for (let i = 0; i < n; i++) {
		scene[i] = new THREE.Scene();
	}
}

function initLight() {
	for (let i = 0; i < n; i++) {
		scene[i].add(new THREE.AmbientLight(0x444444));
		light[i] = new THREE.PointLight(0xffffff);
		light[i].position.set(0, 50, 0);
		//告诉平行光需要开启阴影投射
		light[i].castShadow = true;
		scene[i].add(light[i]);
	}
}

function initModel(path, p) { /////////////////不支持循环赋值
	//辅助工具
	var objLoader = new THREE.OBJLoader();

	if(p==all_pages){
		need_remove = num_last_p; 		// 如果要显示的是最后一页，需要移除的数量就是余数
	}else{
		need_remove = n; 		// 否则，需要移除的数量就是n
	}

	if(p!=all_pages||num_last_p>0){
	objLoader.load(path[n * (p - 1) + 0], function(object) {
		//将模型缩放并 居中 添加到场景当中
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[0].add(object);
	});
	}
	if(p!=all_pages||num_last_p>1){
	objLoader.load(path[n * (p - 1) + 1], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[1].add(object);
	});}
	if(p!=all_pages||num_last_p>2){
	objLoader.load(path[n * (p - 1) + 2], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[2].add(object);
	});}
	if(p!=all_pages||num_last_p>3){
	objLoader.load(path[n * (p - 1) + 3], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[3].add(object);
	});}
	if(p!=all_pages||num_last_p>4){
	objLoader.load(path[n * (p - 1) + 4], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[4].add(object);
	});}
	if(p!=all_pages||num_last_p>5){
	objLoader.load(path[n * (p - 1) + 5], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[5].add(object);
	});}
	if(p!=all_pages||num_last_p>6){
	objLoader.load(path[n * (p - 1) + 6], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[6].add(object);
	});}
	if(p!=all_pages||num_last_p>7){
	objLoader.load(path[n * (p - 1) + 7], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[7].add(object);
	});}
	if(p!=all_pages||num_last_p>8){
	objLoader.load(path[n * (p - 1) + 8], function(object) {
		object.children[0].geometry.computeBoundingBox();
		object.children[0].geometry.center();
		object.scale.set(40, 40, 40);
		scene[8].add(object);
	});}


}


function initControls() {
	for (let i = 0; i < n; i++) {
		controls[i] = new THREE.OrbitControls(camera[i], renderer[i].domElement);
		// 如果使用animate方法时，将此函数删除
		//controls.addEventListener( 'change', render );
		// 使动画循环使用时阻尼或自转 意思是否有惯性
		controls[i].enableDamping = true;
		//动态阻尼系数 就是鼠标拖拽旋转灵敏度
		//controls.dampingFactor = 0.25;
		//是否可以缩放
		controls[i].enableZoom = true;
		//是否自动旋转
		controls[i].autoRotate = true;
		//设置相机距离原点的最远距离
		controls[i].minDistance = 1;
		//设置相机距离原点的最远距离
		controls[i].maxDistance = 200;
		//是否开启右键拖拽
		controls[i].enablePan = true;
	}
}

function render() {
	for (let i = 0; i < n; i++) {
		renderer[i].render(scene[i], camera[i]);
		renderer[i].setSize(350, 200);
	}
}

//窗口变动触发的函数
function onWindowResize() {
	for (let i = 0; i < n; i++) {
		camera[i].aspect = window.innerWidth / window.innerHeight;
		camera[i].updateProjectionMatrix();
	}
	render();
}

function animate() {
	//更新控制器
	render();
	for (let i = 0; i < n; i++) {
		controls[i].update();
	}
	requestAnimationFrame(animate);
}

function removeCube(path, p) {
	for (let i = 0; i < need_remove; i++) {
		var allChildren = scene[i].children;
		var lastObject = allChildren[allChildren.length - 1];
		scene[i].remove(lastObject);
	}
	initModel(path, p);
}

function draw(path, p) {

	if(p>all_pages){
		alert('共'+all_pages+'页');
		p=all_pages;
	}
	$("#page_now_input").val(p);
	$("#page_now_input").attr("title", "该类共"+all_pages+'页'); //设置Next和Previous可用
	$("#Previous_a").attr("class", "page-link"); //设置Next和Previous 可用
	$("#Next_a").attr("class", "page-link");
	if (p <= 1) {
		page_now = 1;
		$("#Previous_a").attr("class", "btn disabled");
	} else if (p >= all_pages) {
		page_now = all_pages;
		$("#Next_a").attr("class", "btn disabled");
	} else {
		page_now = p;
	}
	for (let i = 0; i < 5; i++) { //全部li的状态设置为不acitve
		$("#Page" + (i + 1)).attr("class", "page-item");
	}
	//设置唯一的active
	if(page_now>0||page_now<6){
		$("#Page" + page_now).attr("class", "page-item active");
	}

	if (drawed_flag == false) {
		drawed_flag = true;
		initRender();
		initScene();
		initCamera();
		initLight();
		initModel(path, p);
		initControls();
		animate();
		window.onresize = onWindowResize;
	} else {
		removeCube(path, page_now)
	}
}

