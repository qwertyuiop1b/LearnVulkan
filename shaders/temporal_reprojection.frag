#version 450
layout(location=0) in vec2 inUV; layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std430) readonly buffer Data { vec4 values[]; } data;
void main(){ uint i=min(uint(inUV.x*1024.),1023u); vec4 v=data.values[i]; float c=1.-smoothstep(.008,.018,abs(inUV.y-.2-v.x*.25)); float h=1.-smoothstep(.008,.018,abs(inUV.y-.5-v.y*.25)); float d=1.-smoothstep(.008,.018,abs(inUV.y-.78-v.z*.2)); float r=1.-smoothstep(.008,.018,abs(inUV.y-.92-v.w*.08)); outColor=vec4(vec3(.006,.012,.025)+c*vec3(.1,.9,1)+h*vec3(1,.4,.05)+d*vec3(1,.1,.1)+r*vec3(1,.1,1),1); }
