#version 450
layout(set=0,binding=0,std430)readonly buffer Data{vec4 values[];}data;
layout(push_constant)uniform Push{vec4 p[8];}pc;
layout(location=0)flat out vec3 color;
void main(){uint id=gl_InstanceIndex;vec4 v=data.values[id];float x=fract(float(id)*.618)*1.8-0.9;float y=fract(float(id)*.371)*1.6-.8;vec2 tri[3]=vec2[](vec2(-.012,-.012),vec2(.012,-.012),vec2(0,.018));vec2 pos=vec2(x,y)+tri[gl_VertexIndex]*(1.0-v.y*.12);gl_Position=vec4(pos,0,1);color=mix(vec3(.05,.8,1),vec3(1,.25,.04),v.y/4.0);}
