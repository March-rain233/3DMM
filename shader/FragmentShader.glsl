#version 330 core
 
in vec3 v_Color;
in vec3 v_Normal;
in vec3 v_CamDir;
uniform vec3 u_LightDir; // 定向光方向
uniform vec3 u_LightColor; // 定向光颜色
uniform vec3 u_AmbientColor; // 环境光颜色
uniform float u_Shiny; // 高光系数，非负数，数值越大高光点越小
uniform float u_Specular; // 镜面反射系数，0~1之间的浮点数，影响高光亮度
uniform float u_Diffuse; // 漫反射系数，0~1之间的浮点数，影响表面亮度
uniform float u_Pellucid; // 透光系数，0~1之间的浮点数，影响背面亮
void main() { 
    vec3 lightDir = normalize(-u_LightDir); // 光线向量取反后单位化
    vec3 middleDir = normalize(v_CamDir + lightDir); // 视线和光线的中间向量
    vec4 color = vec4(v_Color, 1);
    float diffuseCos = u_Diffuse * max(0.0, dot(lightDir, v_Normal)); // 光线向量和法向量的内积
    float specularCos = u_Specular * max(0.0, dot(middleDir, v_Normal)); // 中间向量和法向量内
    if (!gl_FrontFacing) 
        diffuseCos *= u_Pellucid; // 背面受透光系数影
    if (diffuseCos == 0.0) 
        specularCos = 0.0;
    else
        specularCos = pow(specularCos, u_Shiny);
    vec3 scatteredLight = min(u_AmbientColor + u_LightColor * diffuseCos, vec3(1.0)); // 散射光
    vec3 reflectedLight = u_LightColor * specularCos; // 反射光
    vec3 rgb = min(color.rgb * (scatteredLight + reflectedLight), vec3(1.0));
    rgb = pow(rgb, vec3(.5));
    gl_FragColor = vec4(rgb, color.a);
} 