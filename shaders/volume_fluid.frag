#version 450
layout(location=0)in vec2 inUV;layout(location=0)out vec4 outColor;layout(set=0,binding=0,std430)readonly buffer Data{vec4 values[];}data;
void main(){uint i=min(uint(inUV.x*1024.),1023u);vec4 v=data.values[i];float a=1.-smoothstep(.008,.018,abs(inUV.y-.18-v.x*.2));float b=1.-smoothstep(.008,.018,abs(inUV.y-.44-v.y*.2));float c=1.-smoothstep(.008,.018,abs(inUV.y-.7-v.z*.2));float d=1.-smoothstep(.008,.018,abs(inUV.y-.9-v.w*.08));outColor=vec4(vec3(.004,.012,.028)+a*vec3(.2,.5,1)+b*vec3(.35,.15,1)+c*vec3(.05,1,.7)+d*vec3(1,.4,.05),1);}
