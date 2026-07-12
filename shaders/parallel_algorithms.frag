#version 450
layout(location=0) in vec2 inUV; layout(location=0) out vec4 outColor;
layout(set=0,binding=0,std430) readonly buffer Data { vec4 values[]; } data;
void main(){ uint i=min(uint(inUV.x*1024.),1023u); vec4 v=data.values[i]; float a=1.-smoothstep(.008,.018,abs(inUV.y-.18-v.x*.23)); float b=1.-smoothstep(.008,.018,abs(inUV.y-.46-v.y*.36)); float c=v.z>=0.?1.-smoothstep(.008,.018,abs(inUV.y-.72-(v.z/2.)*.2)):0.; float d=1.-smoothstep(.008,.018,abs(inUV.y-.9-v.w*.08)); outColor=vec4(vec3(.008,.016,.03)+a*vec3(1,.25,.08)+b*vec3(.08,.8,1)+c*vec3(.35,1,.2)+d*vec3(.9,.2,1),1); }
