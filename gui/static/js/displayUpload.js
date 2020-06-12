function UserinitRender() {
    Userrenderer = new THREE.WebGLRenderer({
        antialias: true
    });
    //告诉渲染器需要阴影效果
    Userrenderer.setClearColor(0x000000);
    document.getElementById('pos').appendChild(Userrenderer.domElement);
}

function UserinitCamera() {
    Usercamera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    Usercamera.position.set(0, 40, 50);
    Usercamera.lookAt(new THREE.Vector3(0, 0, 0));
}

function UserinitScene() {
    Userscene = new THREE.Scene();
}


function UserinitLight() {
    Userscene.add(new THREE.AmbientLight(0x444444));
    Userlight = new THREE.PointLight(0xffffff);
    Userlight.position.set(0, 50, 0);
    //告诉平行光需要开启阴影投射
    Userlight.castShadow = true;
    Userscene.add(Userlight);
}

function UserinitModel(Userpath) {
    //辅助工具
    var Userhelper = new THREE.AxesHelper(50);
    Userscene.add(Userhelper);
    var UserobjLoader = new THREE.OBJLoader();
    //设置当前加载的纹理
    UserobjLoader.load(Userpath, function (object) {
        object.children[0].geometry.computeBoundingBox();
        object.children[0].geometry.center();
        object.scale.set(40, 40, 40);
        Userscene.add(object); //会把所有的模型都添加
    });
}


function UserinitControls() {

    Usercontrols = new THREE.OrbitControls(Usercamera, Userrenderer.domElement);
    // 如果使用animate方法时，将此函数删除
    //controls.addEventListener( 'change', render );
    // 使动画循环使用时阻尼或自转 意思是否有惯性
    Usercontrols.enableDamping = true;
    //动态阻尼系数 就是鼠标拖拽旋转灵敏度
    //controls.dampingFactor = 0.25;
    //是否可以缩放
    Usercontrols.enableZoom = true;
    //是否自动旋转
    Usercontrols.autoRotate = true;
    //设置相机距离原点的最远距离
    Usercontrols.minDistance = 1;
    //设置相机距离原点的最远距离
    Usercontrols.maxDistance = 200;
    //是否开启右键拖拽
    Usercontrols.enablePan = true;
}

function Userrender() {
    Userrenderer.render(Userscene, Usercamera);
    Userrenderer.setSize(400, 300);
}

//窗口变动触发的函数
function UseronWindowResize() {
    Usercamera.aspect = window.innerWidth / window.innerHeight;
    Usercamera.updateProjectionMatrix();
    Userrender();
}

function Useranimate() {
    //更新控制器
    Userrender();
    //更新性能插件
    Usercontrols.update();
    requestAnimationFrame(Useranimate);
}

function UserremoveCube(Userpath) {
    var UserallChildren = Userscene.children;
    var UserlastObject = UserallChildren[UserallChildren.length - 1];
    Userscene.remove(UserlastObject);
    UserinitModel(Userpath);
}

function UsershowModel(Userpath) {
    $(ShowModalLabel).text(Userpath);
    if (userflag == false) {
        $(ShowModalLabel).text(Userpath);
        userflag = true; //设置为场景中已经存在模型
        UserinitRender();
        UserinitScene();
        UserinitCamera();
        UserinitLight();
        UserinitModel(Userpath);
        UserinitControls();
        Useranimate();
        window.onresize = UseronWindowResize;
    } else {
        UserremoveCube(Userpath);
    }
}

function check_before_upload() {
    let filename = $('#up_file_btn').val();
    const obj = filename.lastIndexOf("\\");
    filename = filename.substr(obj + 1);
    if (filename == undefined || filename == '') {
        alert('未上传新文件！');
        return false;
    }
    let fileFormat = filename.substring(filename.lastIndexOf(".")).toLowerCase();
    // 检查是否是图片
    if (fileFormat.match(/.png|.jpg|.jpeg|.bmp/)) {
        $('#location').val(filename);
    $('#i-check').val('重新选择模型或图片');
        return true;
    } else if (fileFormat.match(/.off|.obj|.stl|.3ds|.fbx|.x3d|.ply/)) {
        $('#location').val(filename);
        $('#i-check').val('重新选择模型或图片');
        return true;
    } else {
        alert('上传错误,文件格式必须为：off/obj/stl/3ds/fbx/x3d/ply/png/jpg/jpeg/bmp');
        return false;
    }
}

function check_after_upload(filename) {      //提交后的回调函数

    let tmppath = 'static/tmp/';
    let fileFormat;
    if (!filename) {
        alert('未上传文件');
        $('#i-check').val('上传模型或图片');
        return;
    } else {
        fileFormat = filename.substring(filename.lastIndexOf(".")).toLowerCase();
        // 检查是否是图片
        if (fileFormat.match(/.png|.jpg|.jpeg|.bmp/)) {
            $('#pos').hide();
            $('#posimg').attr('src', tmppath + filename);
            $('#posimg').show();

        } else if (fileFormat.match(/.off|.obj|.stl|.3ds|.fbx|.x3d|.ply/)) {
            $('#posimg').hide();
            filename = filename.split(".")[0]+'.obj';
            UsershowModel(tmppath + filename);
            $('#pos').show();
        } else {
            $("#posimg").hide();
            alert('上传错误,文件格式必须为：off/obj/stl/3ds/fbx/ply/png/jpg/jpeg/bmp');
            return;
        }
    }
}

function check_and_upload() {
    $('#myForm').ajaxSubmit({
            type: 'post',
            url: '/upload',
            beforeSubmit: check_before_upload,  //提交前的回调函数
            success: check_after_upload,      //提交后的回调函数
        }
    )
}
