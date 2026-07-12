#version 450
layout(location=0)in vec2 inUV;layout(location=0)out vec4 outColor;layout(set=0,binding=0,std430)readonly buffer Data{vec4 values[];}data;
void main(){uint i=min(uint(inUV.x*1024.),1023u);vec4 v=data.values[i];float a=1.-smoothstep(.008,.018,abs(inUV.y-.17-v.x*.2));float b=1.-smoothstep(.008,.018,abs(inUV.y-.44-v.y*.08));float c=1.-smoothstep(.008,.018,abs(inUV.y-.7-v.z*.2));float d=1.-smoothstep(.008,.018,abs(inUV.y-.9-v.w*.08));outColor=vec4(vec3(.006,.01,.02)+a*vec3(.1,.8,1)+b*vec3(1,.3,.05)+c*vec3(.2,1,.35)+d*vec3(1,.15,1),1);}
