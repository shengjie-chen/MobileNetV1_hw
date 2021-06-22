#include <iostream>
#define AP_INT_MAX_W 16384
#include "ap_int.h"
#include <iomanip>
#include <hls_stream.h>
#include "top_funtion.hpp"


using namespace hls;
using namespace std;



//#define DEBUG


void do_compute(stream<my_ap_axis >& in, stream<ap_uint<LINEAR_0_OUT_BIT* LINEAR_0_OUT_LEN> >& out, const unsigned int reps) {
#pragma HLS DATAFLOW

	const unsigned int num_per_rep = 224 * 224 * 3 * 8 / 64;

	hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable=in_stream_extract depth=16 dim=1
	ExtractPixels<64, num_per_rep>(in, in_stream_extract, reps);//��my_ap_axis�ṹ������ݶ���64λ��data����8��������Ϣ�������num_per_rep��������

	hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable=in_stream0 depth=16 dim=1
	StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);//���һ������64*3bit

	hls::stream<ap_uint<CONV_0_IN_BIT* CONV_0_IFM_CH> > in_stream1("in_stream1");
#pragma HLS STREAM variable=in_stream1 depth=16 dim=1

	StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT* CONV_0_IFM_CH, num_per_rep / 3>(in_stream0, in_stream1, reps);//���һ�����ݰ���ͼ��ͬһλ������channel���ݣ�ͨ��Ϊ8*3=24



	hls::stream<ap_uint<CONV_0_OUT_BIT* CONV_0_OFM_CH>>  conv_0_out("conv_0_out");
//#pragma HLS STREAM variable=conv_0_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_0_out core=FIFO_LUTRAM
	conv3x3_bn_act_s2 <
		CONV_0_IFM_ROW,
		CONV_0_IFM_COL,
		CONV_0_IFM_CH,
		CONV_0_IN_BIT,

		CONV_0_OFM_CH,
		CONV_0_OUT_BIT,

		CONV_0_W_BIT,
		32,
		CONV_0_INC_BIT,
		CONV_0_BIAS_BIT,

		CONV_0_SIMD,
		CONV_0_PE,
		CONV_0_L_SHIFT>(
			in_stream1,
			conv_0_w,
			conv_0_inc,
			conv_0_bias,
			conv_0_out,
			reps);
#ifdef DEBUG
	cout << "conv_0_out size " << conv_0_out.size() << endl;
#endif

	hls::stream<ap_uint<CONV_1_OUT_BIT* CONV_1_OFM_CH>>  conv_1_out("conv_1_out");
//#pragma HLS STREAM variable=conv_1_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_1_out core=FIFO_LUTRAM
	conv3x3_dw_bn_act_simd9 <
		CONV_1_IFM_ROW,
		CONV_1_IFM_COL,
		CONV_1_IFM_CH,
		CONV_1_IN_BIT,

		CONV_1_OFM_CH,
		CONV_1_OUT_BIT,

		CONV_1_W_BIT,
		32,
		CONV_1_INC_BIT,
		CONV_1_BIAS_BIT,

		CONV_1_SIMD,
		CONV_1_PE,
		CONV_1_L_SHIFT>(
			conv_0_out,
			conv_1_w,
			conv_1_inc,
			conv_1_bias,
			conv_1_out,
			reps);
#ifdef DEBUG
	cout << "conv_1_out size " << conv_1_out.size() << endl;
#endif

	hls::stream<ap_uint<CONV_2_OUT_BIT* CONV_2_OFM_CH>>  conv_2_out("conv_2_out");
//#pragma HLS STREAM variable=conv_2_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_2_out core=FIFO_LUTRAM
	conv1x1_bn_act <
		CONV_2_IFM_ROW,
		CONV_2_IFM_COL,
		CONV_2_IFM_CH,
		CONV_2_IN_BIT,

		CONV_2_OFM_CH,
		CONV_2_OUT_BIT,

		CONV_2_W_BIT,
		32,
		CONV_2_INC_BIT,
		CONV_2_BIAS_BIT,

		CONV_2_SIMD,
		CONV_2_PE,
		CONV_2_L_SHIFT>(
			conv_1_out,
			conv_2_w,
			conv_2_inc,
			conv_2_bias,
			conv_2_out,
			reps);
#ifdef DEBUG
	cout << "conv_2_out size " << conv_2_out.size() << endl;
#endif


    hls::stream<ap_uint<CONV_3_OUT_BIT* CONV_3_OFM_CH>>  conv_3_out("conv_3_out");
//#pragma HLS STREAM variable=conv_3_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_3_out core=FIFO_LUTRAM
    conv3x3_dw_bn_act_s2_simd9 <
        CONV_3_IFM_ROW,
        CONV_3_IFM_COL,
        CONV_3_IFM_CH,
        CONV_3_IN_BIT,

        CONV_3_OFM_CH,
        CONV_3_OUT_BIT,

        CONV_3_W_BIT,
        32,
        CONV_3_INC_BIT,
        CONV_3_BIAS_BIT,

        CONV_3_SIMD,
        CONV_3_PE,
        CONV_3_L_SHIFT>(
            conv_2_out,
            conv_3_w,
            conv_3_inc,
            conv_3_bias,
            conv_3_out,
            reps);
#ifdef DEBUG
    cout << "conv_3_out size " << conv_3_out.size() << endl;
#endif


    hls::stream<ap_uint<CONV_4_OUT_BIT* CONV_4_OFM_CH>>  conv_4_out("conv_4_out");
//#pragma HLS STREAM variable=conv_4_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_4_out core=FIFO_LUTRAM
    conv1x1_bn_act <
        CONV_4_IFM_ROW,
        CONV_4_IFM_COL,
        CONV_4_IFM_CH,
        CONV_4_IN_BIT,

        CONV_4_OFM_CH,
        CONV_4_OUT_BIT,

        CONV_4_W_BIT,
        32,
        CONV_4_INC_BIT,
        CONV_4_BIAS_BIT,

        CONV_4_SIMD,
        CONV_4_PE,
        CONV_4_L_SHIFT>(
            conv_3_out,
            conv_4_w,
            conv_4_inc,
            conv_4_bias,
            conv_4_out,
            reps);
#ifdef DEBUG
    cout << "conv_4_out size " << conv_4_out.size() << endl;
#endif


    hls::stream<ap_uint<CONV_5_OUT_BIT* CONV_5_OFM_CH>>  conv_5_out("conv_5_out");
//#pragma HLS STREAM variable=conv_5_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_5_out core=FIFO_LUTRAM
    conv3x3_dw_bn_act_simd9<
        CONV_5_IFM_ROW,
        CONV_5_IFM_COL,
        CONV_5_IFM_CH,
        CONV_5_IN_BIT,

        CONV_5_OFM_CH,
        CONV_5_OUT_BIT,

        CONV_5_W_BIT,
        32,
        CONV_5_INC_BIT,
        CONV_5_BIAS_BIT,

        CONV_5_SIMD,
        CONV_5_PE,
        CONV_5_L_SHIFT>(
            conv_4_out,
            conv_5_w,
            conv_5_inc,
            conv_5_bias,
            conv_5_out,
            reps);
#ifdef DEBUG
    cout << "conv_5_out size " << conv_5_out.size() << endl;
#endif




    hls::stream<ap_uint<CONV_6_OUT_BIT* CONV_6_OFM_CH>>  conv_6_out("conv_6_out");
//#pragma HLS STREAM variable=conv_6_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_6_out core=FIFO_LUTRAM
    conv1x1_bn_act <
        CONV_6_IFM_ROW,
        CONV_6_IFM_COL,
        CONV_6_IFM_CH,
        CONV_6_IN_BIT,

        CONV_6_OFM_CH,
        CONV_6_OUT_BIT,

        CONV_6_W_BIT,
        32,
        CONV_6_INC_BIT,
        CONV_6_BIAS_BIT,

        CONV_6_SIMD,
        CONV_6_PE,
        CONV_6_L_SHIFT>(
            conv_5_out,
            conv_6_w,
            conv_6_inc,
            conv_6_bias,
            conv_6_out,
            reps);
#ifdef DEBUG
    cout << "conv_6_out size " << conv_6_out.size() << endl;
#endif




    hls::stream<ap_uint<CONV_7_OUT_BIT* CONV_7_OFM_CH>>  conv_7_out("conv_7_out");
//#pragma HLS STREAM variable=conv_7_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_7_out core=FIFO_LUTRAM
    conv3x3_dw_bn_act_s2_simd3_SWU_URAM <
        CONV_7_IFM_ROW,
        CONV_7_IFM_COL,
        CONV_7_IFM_CH,
        CONV_7_IN_BIT,

        CONV_7_OFM_CH,
        CONV_7_OUT_BIT,

        CONV_7_W_BIT,
        32,
        CONV_7_INC_BIT,
        CONV_7_BIAS_BIT,

        CONV_7_SIMD,
        CONV_7_PE,
        CONV_7_L_SHIFT>(
            conv_6_out,
            conv_7_w,
            conv_7_inc,
            conv_7_bias,
            conv_7_out,
            reps);
#ifdef DEBUG
    cout << "conv_7_out size " << conv_7_out.size() << endl;
    // hls::stream<ap_uint<4>> res("res");
    // StreamingDataWidthConverter_Batch<CONV_7_OUT_BIT * CONV_7_OFM_CH, 4, 1>(conv_7_out, res, 1);
    // for (int i=0; i < 64; i ++) {
    //     cout << res.read() << " ";
    // }
    // cout << endl;
    // return;
#endif


    hls::stream<ap_uint<CONV_8_OUT_BIT* CONV_8_OFM_CH>>  conv_8_out("conv_8_out");
//#pragma HLS STREAM variable=conv_8_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_8_out core=FIFO_LUTRAM
    conv1x1_bn_act <
        CONV_8_IFM_ROW,
        CONV_8_IFM_COL,
        CONV_8_IFM_CH,
        CONV_8_IN_BIT,

        CONV_8_OFM_CH,
        CONV_8_OUT_BIT,

        CONV_8_W_BIT,
        32,
        CONV_8_INC_BIT,
        CONV_8_BIAS_BIT,

        CONV_8_SIMD,
        CONV_8_PE,
        CONV_8_L_SHIFT>(
            conv_7_out,
            conv_8_w,
            conv_8_inc,
            conv_8_bias,
            conv_8_out,
            reps);
#ifdef DEBUG
    cout << "conv_8_out size " << conv_8_out.size() << endl;
#endif





    hls::stream<ap_uint<CONV_9_OUT_BIT* CONV_9_OFM_CH>>  conv_9_out("conv_9_out");
//#pragma HLS STREAM variable=conv_9_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_9_out core=FIFO_LUTRAM
    conv3x3_dw_bn_act_s2_simd3_SWU_URAM <
        CONV_9_IFM_ROW,
        CONV_9_IFM_COL,
        CONV_9_IFM_CH,
        CONV_9_IN_BIT,

        CONV_9_OFM_CH,
        CONV_9_OUT_BIT,

        CONV_9_W_BIT,
        32,
        CONV_9_INC_BIT,
        CONV_9_BIAS_BIT,

        CONV_9_SIMD,
        CONV_9_PE,
        CONV_9_L_SHIFT>(
            conv_8_out,
            conv_9_w,
            conv_9_inc,
            conv_9_bias,
            conv_9_out,
            reps);
#ifdef DEBUG
    cout << "conv_9_out size " << conv_9_out.size() << endl;
#endif


    hls::stream<ap_uint<CONV_10_OUT_BIT* CONV_10_OFM_CH>>  conv_10_out("conv_10_out");
//#pragma HLS STREAM variable=conv_10_out depth=108 dim=1
#pragma HLS RESOURCE variable=conv_10_out core=FIFO_LUTRAM
    conv1x1_bn_act <
        CONV_10_IFM_ROW,
        CONV_10_IFM_COL,
        CONV_10_IFM_CH,
        CONV_10_IN_BIT,

        CONV_10_OFM_CH,
        CONV_10_OUT_BIT,

        CONV_10_W_BIT,
        32,
        CONV_10_INC_BIT,
        CONV_10_BIAS_BIT,

        CONV_10_SIMD,
        CONV_10_PE,
        CONV_10_L_SHIFT>(
            conv_9_out,
            conv_10_w,
            conv_10_inc,
            conv_10_bias,
            conv_10_out,
            reps);
#ifdef DEBUG
    cout << "conv_10_out size " << conv_10_out.size() << endl;
#endif


    hls::stream<ap_uint<CONV_11_OUT_BIT* CONV_11_OFM_CH>>  conv_11_out("conv_11_out");
//#pragma HLS STREAM variable=conv_11_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_11_out core=FIFO_LUTRAM
    conv3x3_dw_bn_act_simd3 <
        CONV_11_IFM_ROW,
        CONV_11_IFM_COL,
        CONV_11_IFM_CH,
        CONV_11_IN_BIT,

        CONV_11_OFM_CH,
        CONV_11_OUT_BIT,

        CONV_11_W_BIT,
        32,
        CONV_11_INC_BIT,
        CONV_11_BIAS_BIT,

        CONV_11_SIMD,
        CONV_11_PE,
        CONV_11_L_SHIFT>(
            conv_10_out,
            conv_11_w,
            conv_11_inc,
            conv_11_bias,
            conv_11_out,
            reps);
#ifdef DEBUG
    cout << "conv_11_out size " << conv_11_out.size() << endl;
#endif


    hls::stream<ap_uint<CONV_12_OUT_BIT* CONV_12_OFM_CH>>  conv_12_out("conv_12_out");
//#pragma HLS STREAM variable=conv_12_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_12_out core=FIFO_LUTRAM
    conv1x1_bn_act <
        CONV_12_IFM_ROW,
        CONV_12_IFM_COL,
        CONV_12_IFM_CH,
        CONV_12_IN_BIT,

        CONV_12_OFM_CH,
        CONV_12_OUT_BIT,

        CONV_12_W_BIT,
        32,
        CONV_12_INC_BIT,
        CONV_12_BIAS_BIT,

        CONV_12_SIMD,
        CONV_12_PE,
        CONV_12_L_SHIFT>(
            conv_11_out,
            conv_12_w,
            conv_12_inc,
            conv_12_bias,
            conv_12_out,
            reps);
#ifdef DEBUG
    cout << "conv_12_out size " << conv_12_out.size() << endl;
#endif





    hls::stream<ap_uint<CONV_13_OUT_BIT* CONV_13_OFM_CH>>  conv_13_out("conv_13_out");
    //#pragma HLS STREAM variable=conv_13_out depth=128 dim=1
#pragma HLS RESOURCE variable=conv_13_out core=FIFO_LUTRAM
    conv3x3_dw_bn_act_s2_simd3_SWU_URAM<
		CONV_13_IFM_ROW,
		CONV_13_IFM_COL,
		CONV_13_IFM_CH,
		CONV_13_IN_BIT,

		CONV_13_OFM_CH,
		CONV_13_OUT_BIT,

		CONV_13_W_BIT,
		32,
		CONV_13_INC_BIT,
		CONV_13_BIAS_BIT,

		CONV_13_SIMD,
		CONV_13_PE,
		CONV_13_L_SHIFT>(
			conv_12_out,
			conv_13_w,
			conv_13_inc,
			conv_13_bias,
			conv_13_out,
			reps);
#ifdef DEBUG
	cout << "conv_13_out size " << conv_13_out.size() << endl;
#endif


	hls::stream<ap_uint<CONV_14_OUT_BIT* CONV_14_OFM_CH>>  conv_14_out("conv_14_out");
#pragma HLS RESOURCE variable=conv_14_out core=FIFO_LUTRAM
	conv1x1_bn_act<
		CONV_14_IFM_ROW,
		CONV_14_IFM_COL,
		CONV_14_IFM_CH,
		CONV_14_IN_BIT,

		CONV_14_OFM_CH,
		CONV_14_OUT_BIT,

		CONV_14_W_BIT,
		32,
		CONV_14_INC_BIT,
		CONV_14_BIAS_BIT,

		CONV_14_SIMD,
		CONV_14_PE,
		CONV_14_L_SHIFT>(
			conv_13_out,
			conv_14_w,
			conv_14_inc,
			conv_14_bias,
			conv_14_out,
			reps);
#ifdef DEBUG
	cout << "conv_14_out size " << conv_14_out.size() << endl;
#endif





hls::stream<ap_uint<CONV_14_OFM_CH* LINEAR_0_IN_BIT> > pool_0_out("pool_0_out");
#pragma HLS RESOURCE variable=pool_0_out core=FIFO_LUTRAM
	QuantAvgPool<
		CONV_14_OFM_ROW, CONV_14_OFM_CH, ap_uint<CONV_14_OUT_BIT>,
		2, 5, ap_uint<10>, ap_uint<LINEAR_0_IN_BIT>    >
		(conv_14_out, pool_0_out, reps);
#ifdef DEBUG
	cout << "pool_0_out size " << pool_0_out.size() << endl;
#endif

conv1x1_fclayer<
	1, 1, LINEAR_0_IN_LEN, LINEAR_0_IN_BIT,
	LINEAR_0_OUT_LEN, LINEAR_0_OUT_BIT,
	LINEAR_0_W_BIT, LINEAR_0_OUT_BIT,
	LINEAR_0_SIMD, LINEAR_0_PE, LINEAR_0_L_SHIFT>
	(pool_0_out, linear_0_w, out, reps);


#ifdef DEBUG
	cout << "out size " << out.size() << endl;
#endif



}



void mobile_net(stream<my_ap_axis >& in, stream<ap_uint<LINEAR_0_OUT_BIT* LINEAR_0_OUT_LEN> >& out) {




//#pragma HLS RESOURCE variable=conv_14_w core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=conv_16_w core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=conv_18_w core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=conv_20_w core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=conv_22_w core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=conv_24_w core=ROM_2P_LUTRAM
//#pragma HLS RESOURCE variable=conv_26_w core=ROM_2P_LUTRAM


	//, const unsigned int reps

	/*
	#pragma HLS INTERFACE axis register both port=out
	#pragma HLS INTERFACE axis register both port=in
	#pragma HLS INTERFACE s_axilite port=reps bundle=control
	#pragma HLS INTERFACE s_axilite port=return bundle=control

	#pragma HLS ARRAY_PARTITION variable = conv_0_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_0_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_0_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_1_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_1_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_1_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_2_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_2_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_2_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_3_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_3_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_3_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_4_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_4_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_4_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_5_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_5_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_5_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_6_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_6_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_6_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_7_w complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_7_inc complete dim = 1
	#pragma HLS ARRAY_PARTITION variable = conv_7_bias complete dim = 1

	#pragma HLS ARRAY_PARTITION variable = conv_8_w complete dim = 1
	*/



#pragma HLS ARRAY_PARTITION variable = conv_0_w complete dim = 1
#pragma HLS RESOURCE variable=conv_0_inc core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=conv_0_bias core=ROM_nP_LUTRAM

#pragma HLS RESOURCE variable=conv_1_w core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=conv_1_inc core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=conv_1_bias core=ROM_1P_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = conv_1_w complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_w complete dim = 1
#pragma HLS RESOURCE variable=conv_2_inc core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=conv_2_bias core=ROM_nP_LUTRAM

//#pragma HLS ARRAY_PARTITION variable = conv_3_w complete dim = 1
#pragma HLS RESOURCE variable=conv_3_w core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=conv_3_inc core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=conv_3_bias core=ROM_1P_LUTRAM

#pragma HLS ARRAY_PARTITION variable = conv_4_w complete dim = 1
#pragma HLS RESOURCE variable=conv_4_inc core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=conv_4_bias core=ROM_nP_LUTRAM


//#pragma HLS ARRAY_PARTITION variable = conv_5_w complete dim = 1
#pragma HLS RESOURCE variable=conv_5_w core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=conv_5_inc core=ROM_1P_LUTRAM
#pragma HLS RESOURCE variable=conv_5_bias core=ROM_1P_LUTRAM

#pragma HLS ARRAY_PARTITION variable = conv_6_w complete dim = 1
#pragma HLS RESOURCE variable=conv_6_inc core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=conv_6_bias core=ROM_nP_LUTRAM

#pragma HLS ARRAY_PARTITION variable = conv_7_w complete dim = 1
#pragma HLS RESOURCE variable=conv_7_inc core=ROM_nP_LUTRAM
#pragma HLS RESOURCE variable=conv_7_bias core=ROM_nP_LUTRAM


#pragma HLS ARRAY_PARTITION variable = conv_8_w complete dim = 1





#pragma HLS ARRAY_PARTITION variable = conv_9_w complete dim = 1


#pragma HLS ARRAY_PARTITION variable = conv_10_w complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_11_w complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_12_w complete dim = 1


#pragma HLS ARRAY_PARTITION variable = conv_13_w complete dim = 1


#pragma HLS ARRAY_PARTITION variable = conv_14_w complete dim = 1




#pragma HLS RESOURCE variable=linear_0_w core=ROM_1P_LUTRAM
//#pragma HLS ARRAY_PARTITION variable = linear_0_w complete dim = 1
	const unsigned int reps = 1;
	do_compute(in, out, reps);

}
