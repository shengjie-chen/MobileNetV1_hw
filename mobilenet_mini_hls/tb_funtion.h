#include <iostream>
#include "ap_int.h"
#include <iomanip>
#include <hls_stream.h>
#include <fstream>



// axi data
struct my_ap_axis {
    ap_uint<64> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

template<int MAX_IMAGES,
	int IFMDim,
	int OFMDim,
	int IN_CH,
	int OUT_CH,
	int kernel,
	int stride,
	typename TI,
	typename TO,
	typename TW>
	void conv(
		TI const img[MAX_IMAGES][IFMDim][IFMDim][IN_CH], 
		TW const weights[OUT_CH][IN_CH][kernel][kernel], 
		TO out[MAX_IMAGES][OFMDim][OFMDim][OUT_CH]) 
{
	for (int n = 0; n < MAX_IMAGES; n++)
		for (int x = 0; x < OFMDim; x++)
			for (int y = 0; y < OFMDim; y++)
				for (int h = 0; h < OUT_CH; h++) {
					TO tmp = 0;
					for (int ky = 0; ky < kernel; ky++)
						for (int kx = 0; kx < kernel; kx++)
							for (int w = 0; w < IN_CH; w++) {
								tmp += img[n][y * stride + ky][x * stride + kx][w] * weights[h][w][ky][kx];
							}
					out[n][y][x][h] = tmp;
				}
}

template<int MAX_IMAGES,
    int IFMDim,
    int OFMDim,
    int FMCh,
    int kernel,
    int stride,
    typename TI,
    typename TO,
    typename TW>
    void dwsconv(TI const img[MAX_IMAGES][IFMDim][IFMDim][FMCh], TW const weights[FMCh][kernel][kernel], TO out[MAX_IMAGES][OFMDim][OFMDim][FMCh]) {
    for (int n = 0; n < MAX_IMAGES; n++)
        for (int y = 0; y < OFMDim; y++)
            for (int x = 0; x < OFMDim; x++)
                for (int h = 0; h < FMCh; h++) {
                    TO tmp = 0;
                    for (int ky = 0; ky < kernel; ky++)
                        for (int kx = 0; kx < kernel; kx++) {
                            tmp += img[n][stride * y + ky][stride * x + kx][h] * weights[h][ky][kx];
                        }
                    out[n][y][x][h] = tmp;
                }
}


void load_data(const char* path, char* ptr, unsigned int size)
{
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f)
    {
        std::cout << "no such file,please check the file name!/n";
        exit(0);
    }
    f.read(ptr, size);
    f.close();
}


template <unsigned OutStreamW, unsigned NumLines>
void ExtractPixels(stream<my_ap_axis>& in, stream<ap_uint<OutStreamW>>& out,
    const unsigned reps = 1) {
    my_ap_axis temp;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {
#pragma HLS PIPELINE II = 1
        temp = in.read();
        out.write(temp.data(OutStreamW - 1, 0));
    }
}

template <unsigned int InWidth,   // width of input stream
    unsigned int OutWidth,  // width of output stream
    unsigned int NumInWords // number of input words to process
>
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth>>& in,
    hls::stream<ap_uint<OutWidth>>& out,
    const unsigned int numReps) {
    if (InWidth > OutWidth) {
        // emit multiple output words per input word read
        // CASSERT_DATAFLOW(InWidth % OutWidth == 0);
        const unsigned int outPerIn = InWidth / OutWidth;
        const unsigned int totalIters = NumInWords * outPerIn * numReps;
        unsigned int o = 0;
        ap_uint<InWidth> ei = 0;
        for (unsigned int t = 0; t < totalIters; t++) {

            // read new input word if current out count is zero
            if (o == 0) {
                ei = in.read();
            }
            // pick output word from the rightmost position
            ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
            out.write(eo);
            // shift input to get new output word for next iteration
            ei = ei >> OutWidth;
            // increment written output count
            o++;
            // wraparound indices to recreate the nested loop structure
            if (o == outPerIn) {
                o = 0;
            }
        }
    }
    else if (InWidth == OutWidth) {
        // straight-through copy
        for (unsigned int i = 0; i < NumInWords * numReps; i++) {

            ap_uint<InWidth> e = in.read();
            out.write(e);
        }
    }
    else { // InWidth < OutWidth
     // read multiple input words per output word emitted
     // CASSERT_DATAFLOW(OutWidth % InWidth == 0);
        const unsigned int inPerOut = OutWidth / InWidth;
        const unsigned int totalIters = NumInWords * numReps;
        unsigned int i = 0;
        ap_uint<OutWidth> eo = 0;
        for (unsigned int t = 0; t < totalIters; t++) {

            // read input and shift into output buffer
            ap_uint<InWidth> ei = in.read();
            eo = eo >> InWidth;
            eo(OutWidth - 1, OutWidth - InWidth) = ei;
            // increment read input count
            i++;
            // wraparound logic to recreate nested loop functionality
            if (i == inPerOut) {
                i = 0;
                out.write(eo);
            }
        }
    }
}


template<int MAX_IMAGES,
    int IFMDim,int IN_CH,int IN_BIT,
    int OFMDim,int OUT_CH,int OUT_BIT,
    
    int kernel,
    int stride,
    
    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT,
    typename TI,
    typename TO,
    typename TW,
    typename TINC,
    typename TBIA>
    void conv_test(
        TI const img[MAX_IMAGES][IFMDim][IFMDim][IN_CH],
        TW const weights[OUT_CH][IN_CH][kernel][kernel],
        TO out[MAX_IMAGES][OFMDim][OFMDim][OUT_CH],
        TINC conv_inc[PE][OUT_CH / PE],
        TBIA conv_bias[PE][OUT_CH / PE])
{
    ap_int<M_BIT> TEST_M_0[MAX_IMAGES][OFMDim][OFMDim][OUT_CH];
    
    conv<MAX_IMAGES, IFMDim, OFMDim, IN_CH, OUT_CH, kernel, stride, ap_uint<IN_BIT> >
        (img, weights, TEST_M_0);

    for (int c = 0; c < OUT_CH; c++) {
        for (int y = 0; y < OFMDim; y++) {
            for (int x = 0; x < OFMDim; x++) {
                const unsigned D = 1 << (4 - 1 + IN_BIT + L_SHIFT);

                ap_int<M_BIT> bn_res = TEST_M_0[0][y][x][c] * conv_inc[c % PE][c / PE] + conv_bias[c % PE][c / PE];
                ap_uint<OUT_BIT> res;

                if (bn_res > 0) {
                    bn_res = (bn_res + (D >> 1)) >> (4 - 1 + IN_BIT + L_SHIFT);
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
                out[0][y][x][c] = res;
            }
        }
    }
}



template<int MAX_IMAGES,
    int IFMDim, int IN_CH, int IN_BIT,
    int OFMDim, int OUT_CH, int OUT_BIT,

    int kernel,
    int stride,

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT,
    typename TI,
    typename TO,
    typename TW,
    typename TINC,
    typename TBIA>
    void dwsconv_s1_test(
        TI const img[MAX_IMAGES][IFMDim][IFMDim][IN_CH],
        TW const weights[OUT_CH][kernel][kernel],
        TO out[MAX_IMAGES][OFMDim][OFMDim][OUT_CH],
        TINC conv_inc[PE][OUT_CH / PE],
        TBIA conv_bias[PE][OUT_CH / PE])
{
    ap_uint<IN_BIT> IMAGE_PADDED[MAX_IMAGES][IFMDim + 2][IFMDim + 2][IN_CH];
    ap_int<M_BIT> TEST_M[MAX_IMAGES][OFMDim][OFMDim][OUT_CH];
    
    for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++)
        for (unsigned int oy = 0; oy < IFMDim + 2; oy++)
            for (unsigned int ox = 0; ox < IFMDim + 2; ox++)
                for (unsigned int channel = 0; channel < IN_CH; channel++)
                    IMAGE_PADDED[n_image][oy][ox][channel] = 0;
    for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
        for (unsigned int oy = 0; oy < IFMDim; oy++) {
            for (unsigned int ox = 0; ox < IFMDim; ox++) {
                for (unsigned int channel = 0; channel < IN_CH; channel++)
                {
                    IMAGE_PADDED[n_image][oy + 1][ox + 1][channel] = img[n_image][oy][ox][channel];
                }
            }
        }
    }


    dwsconv
        <MAX_IMAGES, IFMDim + 2, OFMDim, OUT_CH, kernel, stride, ap_uint<IN_BIT> >
        (IMAGE_PADDED, weights, TEST_M);
    for (int c = 0; c < OUT_CH; c++) {
        for (int y = 0; y < OFMDim; y++) {
            for (int x = 0; x < OFMDim; x++) {
                const unsigned D = 1 << (4 - 1 + IN_BIT + L_SHIFT);

                ap_int<M_BIT> bn_res = TEST_M[0][y][x][c] * conv_inc[0][c] + conv_bias[0][c];
                ap_uint<OUT_BIT> res;

                if (bn_res > 0) {
                    bn_res = (bn_res + (D >> 1)) >> (4 - 1 + IN_BIT + L_SHIFT);
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
                out[0][y][x][c] = res;
            }
        }
    }
}

template<int MAX_IMAGES,
    int IFMDim, int IN_CH, int IN_BIT,
    int OFMDim, int OUT_CH, int OUT_BIT,

    int kernel,
    int stride,

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT,
    typename TI,
    typename TO,
    typename TW,
    typename TINC,
    typename TBIA>
    void dwsconv_s2_test(
        TI const img[MAX_IMAGES][IFMDim][IFMDim][IN_CH],
        TW const weights[OUT_CH][kernel][kernel],
        TO out[MAX_IMAGES][OFMDim][OFMDim][OUT_CH],
        TINC conv_inc[PE][OUT_CH / PE],
        TBIA conv_bias[PE][OUT_CH / PE])
{
    ap_uint<IN_BIT> IMAGE_PADDED[MAX_IMAGES][IFMDim + 1][IFMDim + 1][IN_CH];
    ap_int<M_BIT> TEST_M[MAX_IMAGES][OFMDim][OFMDim][OUT_CH];

    for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++)
        for (unsigned int oy = 0; oy < IFMDim + 1; oy++)
            for (unsigned int ox = 0; ox < IFMDim + 1; ox++)
                for (unsigned int channel = 0; channel < IN_CH; channel++)
                    IMAGE_PADDED[n_image][oy][ox][channel] = 0;
    for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
        for (unsigned int oy = 0; oy < IFMDim; oy++) {
            for (unsigned int ox = 0; ox < IFMDim; ox++) {
                for (unsigned int channel = 0; channel < IN_CH; channel++)
                {
                    IMAGE_PADDED[n_image][oy + 1][ox + 1][channel] = img[n_image][oy][ox][channel];
                }
            }
        }
    }


    dwsconv
        <MAX_IMAGES, IFMDim + 1, OFMDim, OUT_CH, kernel, 2, ap_uint<IN_BIT> >
        (IMAGE_PADDED, weights, TEST_M);
    for (int c = 0; c < OUT_CH; c++) {
        for (int y = 0; y < OFMDim; y++) {
            for (int x = 0; x < OFMDim; x++) {
                const unsigned D = 1 << (4 - 1 + IN_BIT + L_SHIFT);

                ap_int<M_BIT> bn_res = TEST_M[0][y][x][c] * conv_inc[0][c] + conv_bias[0][c];
                ap_uint<OUT_BIT> res;

                if (bn_res > 0) {
                    bn_res = (bn_res + (D >> 1)) >> (4 - 1 + IN_BIT + L_SHIFT);
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
                out[0][y][x][c] = res;
            }
        }
    }
}


template<int MAX_IMAGE,
    int IFMDim,
    int OFMDim,
    int FMCh,
    int kernel,
    int stride,
    typename TI,
    typename TO>
    void quant_avgpool(
        TI const img[MAX_IMAGE][IFMDim][IFMDim][FMCh],
        TO out[MAX_IMAGE][OFMDim][OFMDim][FMCh])
{
    for (int n = 0; n < MAX_IMAGE; n++)
        for (int x = 0; x < OFMDim; x++)
            for (int y = 0; y < OFMDim; y++)
                for (int h = 0; h < FMCh; h++) {
                    long long tmp = 0;
                    for (int ky = 0; ky < kernel; ky++)
                        for (int kx = 0; kx < kernel; kx++) {
                            tmp += img[n][y + ky][x + kx][h];
                        }
                    tmp = tmp / 32;
                    out[n][x][y][h] = tmp;
                }
}