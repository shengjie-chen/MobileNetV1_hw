#include <iostream>
#include "ap_int.h"
#include <iomanip>
#include <hls_stream.h>


using namespace hls;
using namespace std;


/**
 * 实现量化激活算法
 * 使用二分查找
 * TODO
 * 丢失精度 暂时弃用
 */
 // int d = 0;
 // template <	unsigned IN_BIT,
 // 			unsigned OUT_BIT,
 // 			unsigned INC_BIT,
 // 			unsigned BIAS_BIT>
 // ap_uint<OUT_BIT> bn_qurelu( ap_int<IN_BIT> in,
 //                 ap_int<INC_BIT> inc,
 //                 ap_int<BIAS_BIT> bias ) {   

 // 	if (d < 16) {
 // 		cout << d << " in " << in << " inc " << inc << " bias " << bias << endl;
 // 		d ++;
 // 	}
 //     ap_int<IN_BIT> target = in + bias;
 //     ap_uint<OUT_BIT> index = 1 << (OUT_BIT - 1);

 // 	// 计算所使用的数据宽度 INC_BIT+OUT_BIT
 // 	ap_int<INC_BIT+OUT_BIT> inc_exp = inc; 
 // 	// 直接对inc移位会溢出 所以初始化时需要 位宽扩展
 //     ap_int<INC_BIT+OUT_BIT + 1> mid = inc_exp << (OUT_BIT - 1);

 //     for(int i=OUT_BIT-2; i >= 0; i --) {
 // #pragma HLS UNROLL
 //         // TODO
 //         // 因为不能特别确定 IN_BIT 和 inc_BIT 关系 所以这里可能有精度损失
 //         ap_int<INC_BIT+OUT_BIT> inc_shift = inc_exp << i;
 //         ap_uint<INC_BIT+OUT_BIT> one_shift = 1 << i;
 //         if(target < mid) {
 //             mid -= inc_shift;
 //             index -= one_shift; 
 //         } else if(mid < target){
 //             mid += inc_shift;
 //             index += one_shift;
 //         }
 //     }
 //     if(target < mid) {
 //         index --;
 //     }
 //     return index;
 // }

 // int d = 0;
template <	unsigned IN_BIT,
    unsigned OUT_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned DATA_BIT,
    unsigned W_BIT,
    unsigned L_SHIFT
>
ap_uint<OUT_BIT> bn_qurelu(ap_int<IN_BIT> in,
    ap_int<INC_BIT> inc,
    ap_int<BIAS_BIT> bias) {

    const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);

    ap_int<IN_BIT> bn_res = in * inc + bias;
    ap_uint<OUT_BIT> res;

    if (bn_res > 0) {
        bn_res = (bn_res + (D >> 1)) >> (W_BIT - 1 + DATA_BIT + L_SHIFT);
        if (bn_res > 15) {
            res = 15;
        }
        else {
            res = bn_res;
        }
    }
    else {
        res = 0;
    }
    return res;

}

//template<   unsigned K,
//    unsigned S,
//    unsigned Din_H,
//    unsigned Din_W,
//    unsigned Cin,
//    unsigned Ibit   >
//    void SWU(
//        stream<ap_uint<Cin* Ibit> >& in,
//        stream<ap_uint<Cin* Ibit> >& out,
//        const unsigned reps = 1)
//{
//    //static_assert((Din_W - K) % S == 0, "(Din_W-K) mod S is not 0");
//    //static_assert((Din_H - K) % S == 0, "(Din_H-K) mod S is not 0");
//    //static_assert(K >= S, "K is not >= than S");
//
//    const unsigned steps = (Din_W - K) / S + 1;
//    const unsigned line_buffer_size = K * Din_W;
//#ifdef SWU_DEBUG
//    cout << "steps: " << steps << endl;
//    cout << "line_buffer_size: " << line_buffer_size << endl;
//#endif
//
//    ap_uint<Cin* Ibit> line_buffer[line_buffer_size];
//
//
//    ap_uint<Cin* Ibit> temp_in;
//
//    ap_uint<1> initial_fill = 0;
//    unsigned stride = 0;
//    unsigned pointer = 0;
//    unsigned h = 0;
//
//    for (unsigned rep = 0; rep < reps * Din_H; rep++) {
//
//        if (h == Din_H) {
//            initial_fill = 0;
//            stride = 0;
//            pointer = 0;
//            h = 0;
//        }
//        h += 1;
//
//#ifdef SWU_DEBUG
//        cout << "wpointer: " << pointer << endl;
//#endif
//
//        for (unsigned w = 0; w < Din_W; w++) {
//
//            temp_in = in.read();
//
//            unsigned line_buffer_pointer = pointer + w;
//            if (line_buffer_pointer >= line_buffer_size) {
//                line_buffer_pointer = line_buffer_pointer - line_buffer_size;
//            }
//#ifdef SWU_DEBUG
//            cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
//#endif
//            line_buffer[line_buffer_pointer] = temp_in;
//        }
//
//        stride += 1;
//        pointer += Din_W;
//        if (pointer >= line_buffer_size) {
//            pointer = pointer - line_buffer_size;
//            initial_fill = 1;
//#ifdef SWU_DEBUG
//            cout << "initial_fill set to 1!" << endl;
//#endif
//        }
//
//#ifdef SWU_DEBUG
//        cout << "stride: " << stride << endl;
//        cout << "rpointer: " << pointer << endl;
//        cout << "line_buffer for out: ";
//        for (unsigned j = 0; j < line_buffer_size; j++) {
//            cout << line_buffer[j] << " ";
//        }
//        cout << endl;
//#endif
//        if (initial_fill == 1 && stride >= S) {
//            stride = 0;
//
//            unsigned s = 0;
//            unsigned x = 0;
//            unsigned y = 0;
//
//            for (unsigned i = 0; i < steps * (K * K); i++) {
//
//                unsigned read_address = (pointer + s * S) + y * Din_W + x;
//
//                if (read_address >= line_buffer_size)
//                    read_address = read_address - line_buffer_size;
//#ifdef SWU_DEBUG
//                cout << "read_address: " << read_address << endl;
//#endif
//                ap_uint<Cin* Ibit> temp_out = line_buffer[read_address];
//                out.write(temp_out);
//
//                if (x == K - 1) {
//                    x = 0;
//                    if (y == K - 1) {
//                        y = 0;
//                        if (s == steps - 1)
//                            s = 0;
//                        else
//                            s++;
//                    }
//                    else
//                        y++;
//                }
//                else
//                    x++;
//            }
//        }
//    }
//}

/**
 *  padding 函数
 */
template <	unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,
    unsigned P>
    void padding_s2(
        // 将每一数竖看成一个元素
        stream<ap_uint<IN_CH* IN_BIT> >& in,
        stream<ap_uint<IN_CH* IN_BIT> >& out,
        const unsigned reps = 1)
{
    const unsigned OUT_ROW = IN_ROW + P;
    const unsigned OUT_COL = IN_COL + P;

    ap_uint<IN_CH* IN_BIT> temp_out = 0;

    for (unsigned rep = 0; rep < reps; rep++) {

        for (unsigned h = 0; h < P; h++) {
            for (unsigned s = 0; s < OUT_COL; s++) {
                out.write(0);
            }
        }

        for (unsigned h = 0; h < IN_ROW; h++) {

            for (unsigned s = 0; s < OUT_COL; s++) {
#pragma HLS PIPELINE II=1

                if (s < P) {
                    temp_out = 0;
                }
                else {
                    temp_out = in.read();
                }
                out.write(temp_out);
            }
        }
    }
}


/**
 *  padding 函数
 */
template <	unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,
    unsigned P>
    void padding(
        // 将每一数竖看成一个元素
        stream<ap_uint<IN_CH* IN_BIT> >& in,
        stream<ap_uint<IN_CH* IN_BIT> >& out,
        const unsigned reps = 1)
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

    ap_uint<IN_CH* IN_BIT> temp_out = 0;

    for (unsigned rep = 0; rep < reps; rep++) {

        for (unsigned h = 0; h < P; h++) {
            for (unsigned s = 0; s < OUT_COL; s++) {
                out.write(0);
            }
        }

        for (unsigned h = 0; h < IN_ROW; h++) {

            for (unsigned s = 0; s < OUT_COL; s++) {
#pragma HLS PIPELINE II=1

                if ((s < P) || (s >= OUT_COL - P)) {
                    temp_out = 0;
                }
                else {
                    temp_out = in.read();
                }

                out.write(temp_out);
            }
        }

        for (unsigned h = 0; h < P; h++) {
            for (unsigned i = 0; i < OUT_COL; i++) {
                out.write(0);
            }
        }

    }
}