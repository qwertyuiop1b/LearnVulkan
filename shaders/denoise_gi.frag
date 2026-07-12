#version 450
layout(location=0)in vec2 inUV;layout(location=0)out vec4 outColor;layout(set=0,binding=0,std430)readonly buffer Data{vec4 values[];}data;
void main(){uint i=min(uint(inUV.x*1024.),1023u);vec4 v=data.values[i];float a=1.-smoothstep(.008,.018,abs(inUV.y-.15-v.x*.18));float b=1.-smoothstep(.008,.018,abs(inUV.y-.4-v.y*.18));float c=1.-smoothstep(.008,.018,abs(inUV.y-.65-v.z*.18));float d=1.-smoothstep(.008,.018,abs(inUV.y-.88-v.w*.1));outColor=vec4(vec3(.006,.012,.025)+a*vec3(1,.08,.04)+b*vec3(.1,1,.4)+c*vec3(.2,.5,1)+d*vec3(1,.5,.08),1);}
