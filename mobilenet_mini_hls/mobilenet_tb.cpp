#include <iomanip>
#include <hls_stream.h>
#include <iostream>
#include <stdint.h>
#define AP_INT_MAX_W 16384
#include "ap_int.h"


using namespace hls;
using namespace std;

#include "tb_funtion.h"
//#include "mobilenet_top.cpp"
#include "param.h"
#include "config.h"

#define MAX_IMAGES 1
#define M_BIT 32
#define LINEAR_0_OUT_BIT 19
#define OFMDim 1


void mobile_net(stream<my_ap_axis >& in, stream<ap_uint<LINEAR_0_OUT_BIT* LINEAR_0_OUT_LEN> >& out);//, const unsigned int reps


int main()
{
	int reps = 1;
	    uint8_t img[224][224][3];
	    load_data("/home/csj/hls_verify/mobilenet_test/data/tulip.bin", (char*)img, sizeof(img));

	    uint8_t* data = (uint8_t*)img;
	    const int data_points_per_line = 8;        // ch * 10
	    const int nums_line_pre_img = 224 * 224 * 3 / 8;

	    hls::stream<my_ap_axis> input_stream("input stream");
	    hls::stream<my_ap_axis> input_stream_dut("input stream_dut");
	    for (unsigned int i = 0; i < nums_line_pre_img; i++) {
	        my_ap_axis temp;
	        for (unsigned int j = 0; j < data_points_per_line; j++) {
	            temp.data(8 * (j + 1) - 1, 8 * j) = data[i * data_points_per_line + j];
	        }
	        input_stream.write(temp);
	        input_stream_dut.write(temp);
	    }

	    const unsigned int num_per_rep = 224 * 224 * 3 * 8 / 64;

	    hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
	    ExtractPixels<64, num_per_rep>(input_stream, in_stream_extract, reps);//\ufffd\ufffdmy_ap_axis\ufffd\u1e79\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\u0776\ufffd\ufffd\ufffd64\u03bb\ufffd\ufffddata\ufffd\ufffd\ufffd\ufffd8\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\u03e2\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffdnum_per_rep\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd

	    hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
	    StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);//\ufffd\ufffd\ufffd\u04bb\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd64*3bit

	    hls::stream<ap_uint<CONV_0_IN_BIT* CONV_0_IFM_CH> > in_stream1("in_stream1");
	    StreamingDataWidthConverter_Batch<64 * 3, CONV_0_IN_BIT* CONV_0_IFM_CH, num_per_rep / 3>(in_stream0, in_stream1, reps);//\ufffd\ufffd\ufffd\u04bb\ufffd\ufffd\ufffd\ufffd\ufffd\u0770\ufffd\ufffd\ufffd\u037c\ufffd\ufffd\u036c\u04bb\u03bb\ufffd\ufffd\ufffd\ufffd\ufffd\ufffdchannel\ufffd\ufffd\ufffd\u0763\ufffd\u0368\ufffd\ufffd\u03aa8*3=24

	    //cout << "in_stream1 size " << in_stream1.size() << endl;

	    hls::stream<ap_uint<CONV_0_IN_BIT* CONV_0_IFM_CH> > in_stream_test("in_stream1_test");
	    ap_uint<CONV_0_IN_BIT* CONV_0_IFM_CH>  temp;
	    for (int i = 0; i < (CONV_0_IFM_ROW * CONV_0_IFM_COL); i++) {
	        temp = in_stream1.read();
	        in_stream_test.write(temp);
	    }

	    std::cout << "Start Verification" << std::endl;



	    ap_uint<CONV_0_IN_BIT> IMAGE_PADDED[MAX_IMAGES][CONV_0_IFM_ROW + 1][CONV_0_IFM_COL + 1][CONV_0_IFM_CH];
	    ap_uint<CONV_0_OUT_BIT> TEST_0[MAX_IMAGES][CONV_0_OFM_ROW][CONV_0_OFM_COL][CONV_0_OFM_CH];

	    for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++)
	        for (unsigned int oy = 0; oy < CONV_0_IFM_ROW + 1; oy++)
	            for (unsigned int ox = 0; ox < CONV_0_IFM_COL + 1; ox++)
	                for (unsigned int channel = 0; channel < CONV_0_IFM_CH; channel++)
	                    IMAGE_PADDED[n_image][oy][ox][channel] = 0;
	    for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
	        for (unsigned int oy = 0; oy < CONV_0_IFM_ROW; oy++) {
	            for (unsigned int ox = 0; ox < CONV_0_IFM_COL; ox++) {
	                ap_uint<CONV_0_IN_BIT* CONV_0_IFM_CH> input_channel = in_stream_test.read();
	                for (unsigned int channel = 0; channel < CONV_0_IFM_CH; channel++)
	                {
	                    ap_uint<CONV_0_IN_BIT> input = input_channel(CONV_0_IN_BIT - 1, 0);
	                    IMAGE_PADDED[n_image][oy + 1][ox + 1][channel] = input;
	                    input_channel = input_channel >> CONV_0_IN_BIT;

	                }
	            }
	        }
	    }
	    ap_int<CONV_0_W_BIT> W_0[CONV_0_OFM_CH][CONV_0_IFM_CH][CONV_0_K][CONV_0_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int oc = 0; oc < CONV_0_OFM_CH; oc++) {
			for (int y = 0; y < CONV_0_K; y++) {
				for (int x = 0; x < CONV_0_K; x++) {
					for (int c = 0; c < CONV_0_IFM_CH; c++) {
						W_0[oc][c][y][x] = conv_0_w[oc % CONV_0_PE][(oc / CONV_0_PE) * (CONV_0_K * CONV_0_K * CONV_0_IFM_CH / CONV_0_SIMD) + CONV_0_K * y + x]((c + 1)* CONV_0_W_BIT - 1, c* CONV_0_W_BIT);
					}
				}
			}
		}
	    conv_test<MAX_IMAGES,
	        CONV_0_IFM_ROW + 1, CONV_0_IFM_CH, CONV_0_IN_BIT,
	        CONV_0_OFM_ROW, CONV_0_OFM_CH, CONV_0_OUT_BIT,
	        CONV_0_K, CONV_0_S,
	        CONV_0_W_BIT, M_BIT, CONV_0_INC_BIT, CONV_0_BIAS_BIT,
	        CONV_0_SIMD, CONV_0_PE, CONV_0_L_SHIFT,
	        ap_uint<CONV_0_IN_BIT>>
			(IMAGE_PADDED, W_0, TEST_0, conv_0_inc, conv_0_bias);
	    std::cout << "Calculation of verification layer 0 is complete" << std::endl;

	    ap_int<CONV_1_W_BIT> W_1[CONV_1_OFM_CH][CONV_1_K][CONV_1_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int c = 0; c < CONV_1_OFM_CH; c++) {
			for (int y = 0; y < CONV_1_K; y++) {
				W_1[c][y][0] = conv_1_w[0][c](3 + y * 12, 0 + y * 12);
				W_1[c][y][1] = conv_1_w[0][c](7 + y * 12, 4 + y * 12);
				W_1[c][y][2] = conv_1_w[0][c](11 + y * 12, 8 + y * 12);
			}
		}
		ap_uint<CONV_1_OUT_BIT> TEST_1[MAX_IMAGES][CONV_1_OFM_ROW][CONV_1_OFM_COL][CONV_1_OFM_CH];
		dwsconv_s1_test<MAX_IMAGES,
			CONV_1_IFM_ROW, CONV_1_IFM_CH, CONV_1_IN_BIT,
			CONV_1_OFM_ROW, CONV_1_OFM_CH, CONV_1_OUT_BIT,
			CONV_1_K, CONV_1_S,
			CONV_1_W_BIT, M_BIT, CONV_1_INC_BIT, CONV_1_BIAS_BIT,
			CONV_1_SIMD, CONV_1_PE, CONV_1_L_SHIFT,
			ap_uint<CONV_1_IN_BIT>>(TEST_0, W_1, TEST_1, conv_1_inc, conv_1_bias);
		std::cout << "Calculation of verification layer 1 is complete" << std::endl;


		ap_uint<CONV_2_OUT_BIT> TEST_2[MAX_IMAGES][CONV_2_OFM_ROW][CONV_2_OFM_COL][CONV_2_OFM_CH];
		ap_int<CONV_2_W_BIT> W_2[CONV_2_OFM_CH][CONV_2_IFM_CH][CONV_2_K][CONV_2_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int oc = 0; oc < CONV_2_OFM_CH; oc++) {
			for (int y = 0; y < CONV_2_K; y++) {
				for (int x = 0; x < CONV_2_K; x++) {
					for (int c = 0; c < CONV_2_IFM_CH; c++) {
						W_2[oc][c][y][x] = conv_2_w[oc % CONV_2_PE][(oc / CONV_2_PE) * (CONV_2_K * CONV_2_K * CONV_2_IFM_CH / CONV_2_SIMD) + (c / CONV_2_SIMD)](((c% CONV_2_SIMD) + 1)* CONV_2_W_BIT - 1, (c% CONV_2_SIMD)* CONV_2_W_BIT);
					}
				}
			}
		}
		conv_test<MAX_IMAGES,
			CONV_2_IFM_ROW, CONV_2_IFM_CH, CONV_2_IN_BIT,
			CONV_2_OFM_ROW, CONV_2_OFM_CH, CONV_2_OUT_BIT,
			CONV_2_K, CONV_2_S,
			CONV_2_W_BIT, M_BIT, CONV_2_INC_BIT, CONV_2_BIAS_BIT,
			CONV_2_SIMD, CONV_2_PE, CONV_2_L_SHIFT,
			ap_uint<CONV_2_IN_BIT>>(TEST_1, W_2, TEST_2, conv_2_inc, conv_2_bias);
		std::cout << "Calculation of verification layer 2 is complete" << std::endl;


		ap_int<CONV_3_W_BIT> W_3[CONV_3_OFM_CH][CONV_3_K][CONV_3_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int c = 0; c < CONV_3_OFM_CH; c++) {
			for (int y = 0; y < CONV_3_K; y++) {
				W_3[c][y][0] = conv_3_w[0][c](3 + y * 12, 0 + y * 12);
				W_3[c][y][1] = conv_3_w[0][c](7 + y * 12, 4 + y * 12);
				W_3[c][y][2] = conv_3_w[0][c](11 + y * 12, 8 + y * 12);
			}
		}
		ap_uint<CONV_3_OUT_BIT> TEST_3[MAX_IMAGES][CONV_3_OFM_ROW][CONV_3_OFM_COL][CONV_3_OFM_CH];
		dwsconv_s2_test<MAX_IMAGES,
			CONV_3_IFM_ROW, CONV_3_IFM_CH, CONV_3_IN_BIT,
			CONV_3_OFM_ROW, CONV_3_OFM_CH, CONV_3_OUT_BIT,
			CONV_3_K, CONV_3_S,
			CONV_3_W_BIT, M_BIT, CONV_3_INC_BIT, CONV_3_BIAS_BIT,
			CONV_3_SIMD, CONV_3_PE, CONV_3_L_SHIFT,
			ap_uint<CONV_3_IN_BIT>>(TEST_2, W_3, TEST_3, conv_3_inc, conv_3_bias);
		std::cout << "Calculation of verification layer 3 is complete" << std::endl;


		ap_uint<CONV_4_OUT_BIT> TEST_4[MAX_IMAGES][CONV_4_OFM_ROW][CONV_4_OFM_COL][CONV_4_OFM_CH];
		ap_int<CONV_4_W_BIT> W_4[CONV_4_OFM_CH][CONV_4_IFM_CH][CONV_4_K][CONV_4_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int oc = 0; oc < CONV_4_OFM_CH; oc++) {
			for (int y = 0; y < CONV_4_K; y++) {
				for (int x = 0; x < CONV_4_K; x++) {
					for (int c = 0; c < CONV_4_IFM_CH; c++) {
						W_4[oc][c][y][x] = conv_4_w[oc % CONV_4_PE][(oc / CONV_4_PE) * (CONV_4_K * CONV_4_K * CONV_4_IFM_CH / CONV_4_SIMD) + (c / CONV_4_SIMD)](((c% CONV_4_SIMD) + 1)* CONV_4_W_BIT - 1, (c% CONV_4_SIMD)* CONV_4_W_BIT);
					}
				}
			}
		}
		conv_test<MAX_IMAGES,
			CONV_4_IFM_ROW, CONV_4_IFM_CH, CONV_4_IN_BIT,
			CONV_4_OFM_ROW, CONV_4_OFM_CH, CONV_4_OUT_BIT,
			CONV_4_K, CONV_4_S,
			CONV_4_W_BIT, M_BIT, CONV_4_INC_BIT, CONV_4_BIAS_BIT,
			CONV_4_SIMD, CONV_4_PE, CONV_4_L_SHIFT,
			ap_uint<CONV_4_IN_BIT>>(TEST_3, W_4, TEST_4, conv_4_inc, conv_4_bias);
		std::cout << "Calculation of verification layer 4 is complete" << std::endl;


		ap_int<CONV_5_W_BIT> W_5[CONV_5_OFM_CH][CONV_5_K][CONV_5_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int c = 0; c < CONV_5_OFM_CH; c++) {
			for (int y = 0; y < CONV_5_K; y++) {
				W_5[c][y][0] = conv_5_w[0][c](3 + y * 12, 0 + y * 12);
				W_5[c][y][1] = conv_5_w[0][c](7 + y * 12, 4 + y * 12);
				W_5[c][y][2] = conv_5_w[0][c](11 + y * 12, 8 + y * 12);
			}
		}
	    ap_uint<CONV_5_OUT_BIT> TEST_5[MAX_IMAGES][CONV_5_OFM_ROW][CONV_5_OFM_COL][CONV_5_OFM_CH];
	    dwsconv_s1_test<MAX_IMAGES,
	        CONV_5_IFM_ROW, CONV_5_IFM_CH, CONV_5_IN_BIT,
	        CONV_5_OFM_ROW, CONV_5_OFM_CH, CONV_5_OUT_BIT,
	        CONV_5_K, CONV_5_S,
	        CONV_5_W_BIT, M_BIT, CONV_5_INC_BIT, CONV_5_BIAS_BIT,
	        CONV_5_SIMD, CONV_5_PE, CONV_5_L_SHIFT,
	        ap_uint<CONV_5_IN_BIT>>(TEST_4, W_5, TEST_5, conv_5_inc, conv_5_bias);
	    std::cout << "Calculation of verification layer 5 is complete" << std::endl;


	    ap_uint<CONV_6_OUT_BIT> TEST_6[MAX_IMAGES][CONV_6_OFM_ROW][CONV_6_OFM_COL][CONV_6_OFM_CH];
	    ap_int<CONV_6_W_BIT> W_6[CONV_6_OFM_CH][CONV_6_IFM_CH][CONV_6_K][CONV_6_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int oc = 0; oc < CONV_6_OFM_CH; oc++) {
	        for (int y = 0; y < CONV_6_K; y++) {
	            for (int x = 0; x < CONV_6_K; x++) {
	                for (int c = 0; c < CONV_6_IFM_CH; c++) {
	                    W_6[oc][c][y][x] = conv_6_w[oc % CONV_6_PE][(oc / CONV_6_PE) * (CONV_6_K * CONV_6_K * CONV_6_IFM_CH / CONV_6_SIMD) + (c / CONV_6_SIMD)](((c% CONV_6_SIMD) + 1)* CONV_6_W_BIT - 1, (c% CONV_6_SIMD)* CONV_6_W_BIT);
	                }
	            }
	        }
	    }
	    conv_test<MAX_IMAGES,
	        CONV_6_IFM_ROW, CONV_6_IFM_CH, CONV_6_IN_BIT,
	        CONV_6_OFM_ROW, CONV_6_OFM_CH, CONV_6_OUT_BIT,
	        CONV_6_K, CONV_6_S,
	        CONV_6_W_BIT, M_BIT, CONV_6_INC_BIT, CONV_6_BIAS_BIT,
	        CONV_6_SIMD, CONV_6_PE, CONV_6_L_SHIFT,
	        ap_uint<CONV_6_IN_BIT>>(TEST_5, W_6, TEST_6, conv_6_inc, conv_6_bias);
	    std::cout << "Calculation of verification layer 6 is complete" << std::endl;


	    ap_int<CONV_7_W_BIT> W_7[CONV_7_OFM_CH][CONV_7_K][CONV_7_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int c = 0; c < CONV_7_OFM_CH; c++) {
	        for (int y = 0; y < CONV_7_K; y++) {
	            W_7[c][y][0] = conv_7_w[0][c * 3 + y](3, 0);
	            W_7[c][y][1] = conv_7_w[0][c * 3 + y](7, 4);
	            W_7[c][y][2] = conv_7_w[0][c * 3 + y](11, 8);
	        }
	    }
	    ap_uint<CONV_7_OUT_BIT> TEST_7[MAX_IMAGES][CONV_7_OFM_ROW][CONV_7_OFM_COL][CONV_7_OFM_CH];
	    dwsconv_s2_test<MAX_IMAGES,
	        CONV_7_IFM_ROW, CONV_7_IFM_CH, CONV_7_IN_BIT,
	        CONV_7_OFM_ROW, CONV_7_OFM_CH, CONV_7_OUT_BIT,
	        CONV_7_K, CONV_7_S,
	        CONV_7_W_BIT, M_BIT, CONV_7_INC_BIT, CONV_7_BIAS_BIT,
	        CONV_7_SIMD, CONV_7_PE, CONV_7_L_SHIFT,
	        ap_uint<CONV_7_IN_BIT>>(TEST_6, W_7, TEST_7, conv_7_inc, conv_7_bias);
	    std::cout << "Calculation of verification layer 7 is complete" << std::endl;


		ap_uint<CONV_8_OUT_BIT> TEST_8[MAX_IMAGES][CONV_8_OFM_ROW][CONV_8_OFM_COL][CONV_8_OFM_CH];
	    ap_int<CONV_8_W_BIT> W_8[CONV_8_OFM_CH][CONV_8_IFM_CH][CONV_8_K][CONV_8_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int oc = 0; oc < CONV_8_OFM_CH; oc++) {
	        for (int y = 0; y < CONV_8_K; y++) {
	            for (int x = 0; x < CONV_8_K; x++) {
	                for (int c = 0; c < CONV_8_IFM_CH; c++) {
	                    W_8[oc][c][y][x] = conv_8_w[oc % CONV_8_PE][(oc / CONV_8_PE) * (CONV_8_K * CONV_8_K * CONV_8_IFM_CH / CONV_8_SIMD) + (c / CONV_8_SIMD)](((c% CONV_8_SIMD) + 1)* CONV_8_W_BIT - 1, (c% CONV_8_SIMD)* CONV_8_W_BIT);
	                }
	            }
	        }
	    }
	    conv_test<MAX_IMAGES,
	        CONV_8_IFM_ROW, CONV_8_IFM_CH, CONV_8_IN_BIT,
	        CONV_8_OFM_ROW, CONV_8_OFM_CH, CONV_8_OUT_BIT,
	        CONV_8_K, CONV_8_S,
	        CONV_8_W_BIT, M_BIT, CONV_8_INC_BIT, CONV_8_BIAS_BIT,
	        CONV_8_SIMD, CONV_8_PE, CONV_8_L_SHIFT,
	        ap_uint<CONV_8_IN_BIT>>(TEST_7, W_8, TEST_8, conv_8_inc, conv_8_bias);
	    std::cout << "Calculation of verification layer 8 is complete" << std::endl;





		ap_int<CONV_9_W_BIT> W_9[CONV_9_OFM_CH][CONV_9_K][CONV_9_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int c = 0; c < CONV_9_OFM_CH; c++) {
	        for (int y = 0; y < CONV_9_K; y++) {
	            W_9[c][y][0] = conv_9_w[0][c * 3 + y](3, 0);
	            W_9[c][y][1] = conv_9_w[0][c * 3 + y](7, 4);
	            W_9[c][y][2] = conv_9_w[0][c * 3 + y](11, 8);
	        }
	    }
	    ap_uint<CONV_9_OUT_BIT> TEST_9[MAX_IMAGES][CONV_9_OFM_ROW][CONV_9_OFM_COL][CONV_9_OFM_CH];
	    dwsconv_s2_test<MAX_IMAGES,
	        CONV_9_IFM_ROW, CONV_9_IFM_CH, CONV_9_IN_BIT,
	        CONV_9_OFM_ROW, CONV_9_OFM_CH, CONV_9_OUT_BIT,
	        CONV_9_K, CONV_9_S,
	        CONV_9_W_BIT, M_BIT, CONV_9_INC_BIT, CONV_9_BIAS_BIT,
	        CONV_9_SIMD, CONV_9_PE, CONV_9_L_SHIFT,
	        ap_uint<CONV_9_IN_BIT>>(TEST_8, W_9, TEST_9, conv_9_inc, conv_9_bias);
	    std::cout << "Calculation of verification layer 9 is complete" << std::endl;


		ap_uint<CONV_10_OUT_BIT> TEST_10[MAX_IMAGES][CONV_10_OFM_ROW][CONV_10_OFM_COL][CONV_10_OFM_CH];
	    ap_int<CONV_10_W_BIT> W_10[CONV_10_OFM_CH][CONV_10_IFM_CH][CONV_10_K][CONV_10_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int oc = 0; oc < CONV_10_OFM_CH; oc++) {
	        for (int y = 0; y < CONV_10_K; y++) {
	            for (int x = 0; x < CONV_10_K; x++) {
	                for (int c = 0; c < CONV_10_IFM_CH; c++) {
	                    W_10[oc][c][y][x] = conv_10_w[oc % CONV_10_PE][(oc / CONV_10_PE) * (CONV_10_K * CONV_10_K * CONV_10_IFM_CH / CONV_10_SIMD) + (c / CONV_10_SIMD)](((c% CONV_10_SIMD) + 1)* CONV_10_W_BIT - 1, (c% CONV_10_SIMD)* CONV_10_W_BIT);
	                }
	            }
	        }
	    }
	    conv_test<MAX_IMAGES,
	        CONV_10_IFM_ROW, CONV_10_IFM_CH, CONV_10_IN_BIT,
	        CONV_10_OFM_ROW, CONV_10_OFM_CH, CONV_10_OUT_BIT,
	        CONV_10_K, CONV_10_S,
	        CONV_10_W_BIT, M_BIT, CONV_10_INC_BIT, CONV_10_BIAS_BIT,
	        CONV_10_SIMD, CONV_10_PE, CONV_10_L_SHIFT,
	        ap_uint<CONV_10_IN_BIT>>(TEST_9, W_10, TEST_10, conv_10_inc, conv_10_bias);
	    std::cout << "Calculation of verification layer 10 is complete" << std::endl;


	    ap_int<CONV_11_W_BIT> W_11[CONV_11_OFM_CH][CONV_11_K][CONV_11_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int c = 0; c < CONV_11_OFM_CH; c++) {
	        for (int y = 0; y < CONV_11_K; y++) {
	            W_11[c][y][0] = conv_11_w[0][c * 3 + y](3, 0);
	            W_11[c][y][1] = conv_11_w[0][c * 3 + y](7, 4);
	            W_11[c][y][2] = conv_11_w[0][c * 3 + y](11, 8);
	        }
	    }
	    ap_uint<CONV_11_OUT_BIT> TEST_11[MAX_IMAGES][CONV_11_OFM_ROW][CONV_11_OFM_COL][CONV_11_OFM_CH];
	    dwsconv_s1_test<MAX_IMAGES,
	        CONV_11_IFM_ROW, CONV_11_IFM_CH, CONV_11_IN_BIT,
	        CONV_11_OFM_ROW, CONV_11_OFM_CH, CONV_11_OUT_BIT,
	        CONV_11_K, CONV_11_S,
	        CONV_11_W_BIT, M_BIT, CONV_11_INC_BIT, CONV_11_BIAS_BIT,
	        CONV_11_SIMD, CONV_11_PE, CONV_11_L_SHIFT,
	        ap_uint<CONV_11_IN_BIT>>(TEST_10, W_11, TEST_11, conv_11_inc, conv_11_bias);
	    std::cout << "Calculation of verification layer 11 is complete" << std::endl;


		ap_uint<CONV_12_OUT_BIT> TEST_12[MAX_IMAGES][CONV_12_OFM_ROW][CONV_12_OFM_COL][CONV_12_OFM_CH];
	    ap_int<CONV_12_W_BIT> W_12[CONV_12_OFM_CH][CONV_12_IFM_CH][CONV_12_K][CONV_12_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
	    for (int oc = 0; oc < CONV_12_OFM_CH; oc++) {
	        for (int y = 0; y < CONV_12_K; y++) {
	            for (int x = 0; x < CONV_12_K; x++) {
	                for (int c = 0; c < CONV_12_IFM_CH; c++) {
	                    W_12[oc][c][y][x] = conv_12_w[oc % CONV_12_PE][(oc / CONV_12_PE) * (CONV_12_K * CONV_12_K * CONV_12_IFM_CH / CONV_12_SIMD) + (c / CONV_12_SIMD)](((c% CONV_12_SIMD) + 1)* CONV_12_W_BIT - 1, (c% CONV_12_SIMD)* CONV_12_W_BIT);
	                }
	            }
	        }
	    }
	    conv_test<MAX_IMAGES,
	        CONV_12_IFM_ROW, CONV_12_IFM_CH, CONV_12_IN_BIT,
	        CONV_12_OFM_ROW, CONV_12_OFM_CH, CONV_12_OUT_BIT,
	        CONV_12_K, CONV_12_S,
	        CONV_12_W_BIT, M_BIT, CONV_12_INC_BIT, CONV_12_BIAS_BIT,
	        CONV_12_SIMD, CONV_12_PE, CONV_12_L_SHIFT,
	        ap_uint<CONV_12_IN_BIT>>(TEST_11, W_12, TEST_12, conv_12_inc, conv_12_bias);
	    std::cout << "Calculation of verification layer 12 is complete" << std::endl;



		ap_int<CONV_13_W_BIT> W_13[CONV_13_OFM_CH][CONV_13_K][CONV_13_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int c = 0; c < CONV_13_OFM_CH; c++) {
		    for (int y = 0; y < CONV_13_K; y++) {
		        W_13[c][y][0] = conv_13_w[0][c * 3 + y](3, 0);
		        W_13[c][y][1] = conv_13_w[0][c * 3 + y](7, 4);
		        W_13[c][y][2] = conv_13_w[0][c * 3 + y](11, 8);
		    }
		}
		ap_uint<CONV_13_OUT_BIT> TEST_13[MAX_IMAGES][CONV_13_OFM_ROW][CONV_13_OFM_COL][CONV_13_OFM_CH];
		dwsconv_s2_test<MAX_IMAGES,
		    CONV_13_IFM_ROW, CONV_13_IFM_CH, CONV_13_IN_BIT,
		    CONV_13_OFM_ROW, CONV_13_OFM_CH, CONV_13_OUT_BIT,
		    CONV_13_K, CONV_13_S,
		    CONV_13_W_BIT, M_BIT, CONV_13_INC_BIT, CONV_13_BIAS_BIT,
		    CONV_13_SIMD, CONV_13_PE, CONV_13_L_SHIFT,
		    ap_uint<CONV_13_IN_BIT>>(TEST_12, W_13, TEST_13, conv_13_inc, conv_13_bias);
		std::cout << "Calculation of verification layer 13 is complete" << std::endl;


		ap_uint<CONV_14_OUT_BIT> TEST_14[MAX_IMAGES][CONV_14_OFM_ROW][CONV_14_OFM_COL][CONV_14_OFM_CH];
		ap_int<CONV_14_W_BIT> W_14[CONV_14_OFM_CH][CONV_14_IFM_CH][CONV_14_K][CONV_14_K];//\u0228\ufffd\u063e\ufffd\ufffd\ufffd
		for (int oc = 0; oc < CONV_14_OFM_CH; oc++) {
		    for (int y = 0; y < CONV_14_K; y++) {
		        for (int x = 0; x < CONV_14_K; x++) {
		            for (int c = 0; c < CONV_14_IFM_CH; c++) {
		                W_14[oc][c][y][x] = conv_14_w[oc % CONV_14_PE][(oc / CONV_14_PE) * (CONV_14_K * CONV_14_K * CONV_14_IFM_CH / CONV_14_SIMD) + (c / CONV_14_SIMD)](((c% CONV_14_SIMD) + 1)* CONV_14_W_BIT - 1, (c% CONV_14_SIMD)* CONV_14_W_BIT);
		            }
		        }
		    }
		}
		conv_test<MAX_IMAGES,
		    CONV_14_IFM_ROW, CONV_14_IFM_CH, CONV_14_IN_BIT,
		    CONV_14_OFM_ROW, CONV_14_OFM_CH, CONV_14_OUT_BIT,
		    CONV_14_K, CONV_14_S,
		    CONV_14_W_BIT, M_BIT, CONV_14_INC_BIT, CONV_14_BIAS_BIT,
		    CONV_14_SIMD, CONV_14_PE, CONV_14_L_SHIFT,
		    ap_uint<CONV_14_IN_BIT>>(TEST_13, W_14, TEST_14, conv_14_inc, conv_14_bias);
		std::cout << "Calculation of verification layer 14 is complete" << std::endl;


//		ap_uint<CONV_14_OUT_BIT> temp_int14;
//				for (int y = 0; y < 7; y++) {
//					for (int x = 0; x < 7; x++) {
//						for (int c = 0; c < CONV_14_OFM_CH; c++) {
//							temp_int14 = TEST_14[0][y][x][c];
//							std::cout << "Expected temp_int14[" << y << "][" << x << "][" << c << "]=" << temp_int14 << std::endl;
//						}
//					}
//				}

		ap_uint<LINEAR_0_IN_BIT> TEST_15[MAX_IMAGES][1][1][LINEAR_0_IN_LEN];
		quant_avgpool<MAX_IMAGES, CONV_14_OFM_ROW, 1, LINEAR_0_IN_LEN, 7, 1, ap_uint<CONV_14_OUT_BIT> >
			(TEST_14, TEST_15);
		std::cout << "Calculation of verification layer 15 is complete" << std::endl;

//		ap_uint<LINEAR_0_IN_BIT> temp_int15;
//		for (int y = 0; y < 1; y++) {
//			for (int x = 0; x < 1; x++) {
//				for (int c = 0; c < LINEAR_0_IN_LEN; c++) {
//					temp_int15 = TEST_15[0][y][x][c];
//					std::cout << "Expected temp_int15[" << y << "][" << x << "][" << c << "]=" << temp_int15 << std::endl;
//				}
//			}
//		}


		ap_uint<LINEAR_0_OUT_BIT> TEST_16[MAX_IMAGES][OFMDim][OFMDim][LINEAR_0_OUT_LEN];
		ap_int<LINEAR_0_W_BIT> W_16[LINEAR_0_OUT_LEN][LINEAR_0_IN_LEN][1][1];//\u0216\ufffd\u063e\ufffd\ufffd\ufffd
		for (int oc = 0; oc < LINEAR_0_OUT_LEN; oc++) {
		    for (int y = 0; y < OFMDim; y++) {
		        for (int x = 0; x < OFMDim; x++) {
		            for (int c = 0; c < LINEAR_0_IN_LEN; c++) {
		                W_16[oc][c][y][x] = linear_0_w[oc % LINEAR_0_PE][(oc / LINEAR_0_PE) * (OFMDim * OFMDim * LINEAR_0_IN_LEN / LINEAR_0_SIMD) + (c / LINEAR_0_SIMD)](((c% LINEAR_0_SIMD) + 1)* LINEAR_0_W_BIT - 1, (c% LINEAR_0_SIMD)* LINEAR_0_W_BIT);
		            }
		        }
		    }
		}
		conv
		    <MAX_IMAGES, OFMDim, OFMDim, LINEAR_0_IN_LEN, LINEAR_0_OUT_LEN, 1, 1, ap_uint<LINEAR_0_IN_BIT> >
		    (TEST_15, W_16, TEST_16);
		std::cout << "Calculation of verification layer 16 is complete" << std::endl;






	    hls::stream<ap_uint<LINEAR_0_OUT_BIT* LINEAR_0_OUT_LEN>>  mobilenet_out("mobilenet_out");
	    mobile_net(input_stream_dut, mobilenet_out);
	    std::cout << "DUT Calculation Complete & Start To Compare" << std::endl;

	    ap_int<LINEAR_0_OUT_BIT> temp_int;
	    ap_uint<LINEAR_0_OUT_BIT> temp_dut;
	    ap_uint<LINEAR_0_OUT_BIT* LINEAR_0_OUT_LEN> temp0;
	    for (int y = 0; y < 1; y++) {
	        for (int x = 0; x < 1; x++) {
	            temp0 = mobilenet_out.read();
	            for (int c = 0; c < LINEAR_0_OUT_LEN; c++) {
	                temp_dut = temp0(LINEAR_0_OUT_BIT * (c + 1) - 1, LINEAR_0_OUT_BIT * c);
	                temp_int = temp_dut;
	                std::cout << temp_int << std::endl;
	                if (temp_dut != TEST_16[0][y][x][c]) {
	                    std::cout << "ERROR: Expected[" << y << "][" << x << "][" << c << "]=" << TEST_16[0][y][x][c] << " actual " << temp_dut << std::endl;
	                    return 1;
	                    //cout << "adj_out[" << y << "][" << x << "][" << j << "][" << i << "] = " << hex << temp_dut << endl;
	                }
	            }
	        }
	    }
	    std::cout << "*********************************************Verification Success*******************************************" << std::endl;

    


}
