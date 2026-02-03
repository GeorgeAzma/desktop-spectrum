struct VSIn { 
    float3 pos : POSITION; 
    float3 col : COLOR; 
};

struct PSIn { 
    float4 pos : SV_POSITION; 
    float3 col : COLOR; 
};

PSIn vs_main(VSIn v) { 
    PSIn o; 
    o.pos = float4(v.pos, 1); 
    o.col = v.col; 
    return o; 
}

float4 ps_main(PSIn i) : SV_Target { 
    return float4(i.col, 1); 
}
   