#include <iostream>
#include "ap_int.h"
#include <iomanip>
#include <hls_stream.h>
#include "funtion.h"
#include "sliding_window_unit.h"
#include "param.h"
#include "config.h"
//#include "matrix_vector_unit.h"
#define LINEAR_0_OUT_BIT 19

using namespace hls;
using namespace std;


// axi data
struct my_ap_axis {
    ap_uint<64> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

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



/**
 * \brief   Accumulate-pool - like average pooling over the whole frame, but without the division at end
 *
 * \tparam ImgDim       Width and Heigth of the Input Feature Map (assumed square)
 * \tparam NumChannels  Number of Input Feature Maps
 * \tparam ActType      DataType of the input activation (as used in the comparison)
 * \tparam PECount      PE parallelism to apply ReLU
 * \tparam AccType      Datatype of the accumulation (e.g. output)
 *
 * \param in            Input stream
 * \param out           Output stream
 * \param numReps       Number of time the function has to be repeatedly executed (e.g. number of images)
 *
 *
 * ����λ��������������ƽ���ػ�
 */
template<
    unsigned int ImgDim,
    unsigned int NumChannels,
    typename ActType,
    unsigned int PECount,
    unsigned int ShiftNum,
    typename AccType,
    typename OutType>
    void QuantAvgPool_Batch(
        stream<ap_uint<PECount* ActType::width> >& in,
        stream<ap_uint<PECount* OutType::width> >& out,
        const unsigned int numReps)
{
    ap_uint<PECount* ActType::width> thin;
    ap_uint<PECount* AccType::width> accumulators[NumChannels / PECount];
    ap_uint<PECount* OutType::width> outbuffer[NumChannels / PECount];

#pragma HLS RESOURCE variable=accumulators core=RAM_2P_LUTRAM

    //call to thresholding library function
    for (unsigned int reps = 0; reps < numReps; reps++) {
        for (unsigned int pixel = 0; pixel < ImgDim * ImgDim; pixel++) {
            for (unsigned int fold = 0; fold < NumChannels / PECount; fold++) {
#pragma HLS PIPELINE II=1
                thin = in.read();
                ap_uint<PECount* AccType::width> accbank = accumulators[fold];
                for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
                    // Threshold and assign to right bits of output buffers
                    unsigned int lowBit = pe * ActType::width;
                    unsigned int highBit = (pe + 1) * ActType::width - 1;
                    ActType val = thin((pe + 1) * ActType::width - 1, pe * ActType::width);
                    AccType acc = accbank((pe + 1) * AccType::width - 1, pe * AccType::width);
                    AccType result;
                    if (pixel == 0)
                        result = val;
                    else
                        result = val + acc;
                    if (pixel == ImgDim * ImgDim - 1) {
                        accbank((pe + 1) * OutType::width - 1, pe * OutType::width) = result >> ShiftNum;
                    }
                    else {
                        accbank((pe + 1) * AccType::width - 1, pe * AccType::width) = result;
                    }

                }
                if (pixel == ImgDim * ImgDim - 1)
                    outbuffer[fold] = accbank;
                else
                    accumulators[fold] = accbank;
            }
        }

        for (unsigned int fold = 0; fold < NumChannels / PECount; fold++)
        {
            out.write(outbuffer[fold]);
        }
    }
}



/**
 *  simd ��
 *  �� �������Զ�ѡ��ʹ�� dsp ���� lut
 */
template <	unsigned W_BIT,
            unsigned IN_BIT,
            unsigned M_BIT,
            unsigned SIMD>
ap_int<M_BIT> simd_mul(
    ap_uint<SIMD* W_BIT> weights,
    ap_uint<SIMD* IN_BIT> in)
{
    ap_int<M_BIT> accumulation = 0;

    for (unsigned p = 0; p < SIMD; p++) {
#pragma HLS UNROLL
        ap_int<W_BIT> temp_w = weights((p + 1) * W_BIT - 1, p * W_BIT);
        ap_uint<IN_BIT> temp_in = in((p + 1) * IN_BIT - 1, p * IN_BIT);
        ap_int<W_BIT + IN_BIT> result = temp_w * temp_in;
        // #pragma HLS RESOURCE variable=result core=Mul_LUT
        accumulation += result;
    }
    return accumulation;
}





/**
 * �����������㵥Ԫ
 * ͬʱ�������������
 */
template <	unsigned MAT_ROW,		// չ�����k �� k �� in_ch
    unsigned MAT_COL,		// չ�����out_ch

    unsigned IN_BIT,
    unsigned OUT_BIT,		// 

    unsigned W_BIT,
    unsigned M_BIT,			// ���ۼӺ�ļ�������ֵ

    unsigned INC_BIT,		// ����Ȳ����� �Ĳ���
    unsigned BIAS_BIT,		// 

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT,
    unsigned VECT_NUMS>
    void matrix_vector_act_unit(
        stream<ap_uint<SIMD* IN_BIT> >& vec,
        const ap_uint<SIMD* W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
        const ap_int<INC_BIT> inc[PE][MAT_COL / PE],
        const ap_int<BIAS_BIT> bias[PE][MAT_COL / PE],
        stream<ap_uint<PE* OUT_BIT> >& out,
        const unsigned reps = 1)
{
    static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
    static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

    const unsigned INPUT_FOLD = MAT_ROW / SIMD;
    const unsigned OUTPUT_FOLD = MAT_COL / PE;

    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;


    // ��Ҫ����һ������
    ap_uint<SIMD* IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_BRAM

    // ���������ۼӽ��
    // ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
    unsigned in_fold_cnt = 0;			// �����۵�����
    unsigned out_fold_cnt = 0;			// ����۵�����
    unsigned tile = 0;

    // һ�� ��������� ��Ҫ���� in_ch * k * k���ȵ�����
    ap_uint<SIMD* IN_BIT> temp_vec;
    // �ۼӽ�� ������Ҫ��ʼ��Ϊ0 
    ap_int<M_BIT> acc[PE];

    // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
    for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1

        // �������ڵ�һ�����֮ǰ �Ͷ��������ݣ�֮��һֱ��
        // ������۵���һ�μ���ʱ��
        if (out_fold_cnt == 0) {
            temp_vec = vec.read();
            row_store[in_fold_cnt] = temp_vec;
        }
        else {
            temp_vec = row_store[in_fold_cnt];
        }

        // index = wVec*OutputFold+wMat;

        // ��ʼ���ۼӽ��
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        // ��Ҫ���㵥Ԫ ������UNROLLչ�� �����õ�����ʵ�ּ���
        // PE ���м���
        for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
            // �� W �ӿ�
            ap_uint<SIMD* W_BIT> temp_mat = weights[p][tile];
            // SIMD ����
            acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
            // if (p == 0)
            // 	cout << temp_vec(7, 0) << " " <<  temp_vec(15, 8) << " " << temp_vec(23, 16) << endl;
        }

        // �����߼� ���������
        tile++;
        if (++in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_uint<PE* M_BIT> out_buf;
            // PE �м������ �������
            for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
                out_buf((p + 1) * OUT_BIT - 1, p * OUT_BIT) = bn_qurelu<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT, L_SHIFT>(acc[p], inc[p][out_fold_cnt], bias[p][out_fold_cnt]);
                // cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " " << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
                // acc[p] = 0;
            }
            out.write(out_buf);
            // ������һ�ξ�����������
            if (++out_fold_cnt == OUTPUT_FOLD) {
                out_fold_cnt = 0;
                tile = 0;
            }

        }
    }  // end for

}


/**
    * �����������㵥Ԫ
    * ���������������
    */
template <	unsigned MAT_ROW,		// չ�����k �� k �� in_ch
    unsigned MAT_COL,		// չ�����out_ch

    unsigned IN_BIT,
    unsigned OUT_BIT,		// 

    unsigned W_BIT,
    unsigned M_BIT,			// ���ۼӺ�ļ�������ֵ

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT,
    unsigned VECT_NUMS>
    void matrix_vector_noact_unit(
        stream<ap_uint<SIMD* IN_BIT> >& vec,
        const ap_uint<SIMD* W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
        stream<ap_uint<PE* OUT_BIT> >& out,
        const unsigned reps = 1)
{
    static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
    static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

    const unsigned INPUT_FOLD = MAT_ROW / SIMD;
    const unsigned OUTPUT_FOLD = MAT_COL / PE;

    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;


    // ��Ҫ����һ������
    ap_uint<SIMD* IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_URAM

    // ���������ۼӽ��
    // ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
    unsigned in_fold_cnt = 0;			// �����۵�����
    unsigned out_fold_cnt = 0;			// ����۵�����
    unsigned tile = 0;

    // һ�� ��������� ��Ҫ���� in_ch * k * k���ȵ�����
    ap_uint<SIMD* IN_BIT> temp_vec;
    // �ۼӽ�� ������Ҫ��ʼ��Ϊ0 
    ap_int<M_BIT> acc[PE];

    // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
    for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1

        // �������ڵ�һ�����֮ǰ �Ͷ��������ݣ�֮��һֱ��
        // ������۵���һ�μ���ʱ��
        if (out_fold_cnt == 0) {
            temp_vec = vec.read();
            row_store[in_fold_cnt] = temp_vec;
        }
        else {
            temp_vec = row_store[in_fold_cnt];
        }

        // index = wVec*OutputFold+wMat;

        // ��ʼ���ۼӽ��
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        // ��Ҫ���㵥Ԫ ������UNROLLչ�� �����õ�����ʵ�ּ���
        // PE ���м���
        for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
            // �� W �ӿ�
            ap_uint<SIMD* W_BIT> temp_mat = weights[p][tile];
            // SIMD ����
            acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
            // if (p == 0)
            // 	cout << temp_vec(7, 0) << " " <<  temp_vec(15, 8) << " " << temp_vec(23, 16) << endl;
        }

        // �����߼� ���������
        tile++;
        if (++in_fold_cnt == INPUT_FOLD) {
            in_fold_cnt = 0;
            ap_uint<PE* M_BIT> out_buf;
            // PE �м������ �������
            for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
                out_buf((p + 1) * OUT_BIT - 1, p * OUT_BIT) = acc[p];
                // cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " " << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
                // acc[p] = 0;
            }
            out.write(out_buf);
            // ������һ�ξ�����������
            if (++out_fold_cnt == OUTPUT_FOLD) {
                out_fold_cnt = 0;
                tile = 0;
            }

        }
    }  // end for

}


/**
    * �����������㵥Ԫ
    * ͬʱ�������������
    * ��������Ⱦ��
    */
template <	unsigned MAT_ROW,		// չ�����k �� k 
    unsigned MAT_COL,		// չ�����out_ch

    unsigned IN_BIT,
    unsigned OUT_BIT,

    unsigned W_BIT,
    unsigned M_BIT,			// ���ۼӺ�ļ�������ֵ

    unsigned INC_BIT,		// ����Ȳ����� �Ĳ���
    unsigned BIAS_BIT,		// 

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT,
    unsigned VECT_NUMS>
    void matrix_vector_act_unit_dw(
        stream<ap_uint<SIMD* IN_BIT> >& vec,
        const ap_uint<SIMD* W_BIT> weights[PE][(MAT_ROW / SIMD) * (MAT_COL / PE)],
        const ap_int<INC_BIT> inc[PE][MAT_COL / PE],
        const ap_int<BIAS_BIT> bias[PE][MAT_COL / PE],
        stream<ap_uint<PE* OUT_BIT> >& out,
        const unsigned reps = 1)
{
    static_assert(MAT_ROW % SIMD == 0, "MAT_ROW mod SIMD is not 0");
    static_assert(MAT_COL % PE == 0, "MAT_COL mod PE is not 0");

    const unsigned INPUT_FOLD = MAT_ROW / SIMD;
    const unsigned OUTPUT_FOLD = MAT_COL / PE;

    const unsigned total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;


    // ��Ҫ����һ������
    ap_uint<SIMD* IN_BIT> row_store[INPUT_FOLD];
#pragma HLS RESOURCE variable=row_store core=RAM_2P_LUTRAM

    // ���������ۼӽ��
    // ap_uint<M_BIT> result_vec[PE];
// #pragma HLS ARRAY_PARTITION variable=result_vec complete dim=0
    unsigned in_fold_cnt = 0;			// �����۵�����
    unsigned out_fold_cnt = 0;			// ����۵�����
    unsigned tile = 0;

    // һ�� ��������� ��Ҫ���� in_ch * k * k���ȵ�����
    ap_uint<SIMD* IN_BIT> temp_vec;
    // �ۼӽ�� ������Ҫ��ʼ��Ϊ0 
    ap_int<M_BIT> acc[PE];

    // total_reps = INPUT_FOLD * OUTPUT_FOLD * VECT_NUMS * reps;
    for (unsigned rep = 0; rep < total_reps; rep++) {
#pragma HLS PIPELINE II=1

        // �������ڵ�һ�����֮ǰ �Ͷ��������ݣ�֮��һֱ��
        // ������۵���һ�μ���ʱ��
        //if (out_fold_cnt == 0) {
        temp_vec = vec.read();
        //	row_store[in_fold_cnt] = temp_vec;
        //}
        //else {
        //	temp_vec = row_store[in_fold_cnt];
        //}

        // index = wVec*OutputFold+wMat;

        // ��ʼ���ۼӽ��
        if (in_fold_cnt == 0) {
            for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
                acc[p] = 0;
            }
        }

        // ��Ҫ���㵥Ԫ ������UNROLLչ�� �����õ�����ʵ�ּ���
        // PE ���м���
        for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
            // �� W �ӿ�
            ap_uint<SIMD* W_BIT> temp_mat = weights[p][tile];
            // SIMD ����
            acc[p] += simd_mul<W_BIT, IN_BIT, M_BIT, SIMD>(temp_mat, temp_vec);
            // if (p == 0)
            // 	cout << temp_vec(7, 0) << " " <<  temp_vec(15, 8) << " " << temp_vec(23, 16) << endl;
        }

        // �����߼� ���������
        tile++;
        if (++in_fold_cnt == INPUT_FOLD)
        {
            in_fold_cnt = 0;
            ap_uint<PE* OUT_BIT> out_buf;
            //std::cout << PE << "  " << OUT_BIT << "  " << PE * OUT_BIT << std::endl;
            // PE �м������ �������
            for (unsigned p = 0; p < PE; p++) {
#pragma HLS UNROLL
                out_buf = bn_qurelu<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT, L_SHIFT>(acc[p], inc[p][out_fold_cnt], bias[p][out_fold_cnt]);
                //out_buf((p + 1) * OUT_BIT - 1, p * OUT_BIT)
                // cout << acc[p] << " " << out_buf((p+1)*OUT_BIT-1, p*OUT_BIT) << " " << inc[p][out_fold_cnt] << " " << bias[p][out_fold_cnt] << "     ";
                // acc[p] = 0;
            }
            out.write(out_buf);
            // ������һ�ξ�����������
            if (++out_fold_cnt == OUTPUT_FOLD) {
                out_fold_cnt = 0;
                tile = 0;
            }
        }
    }  // end for
}


// ����ͬchannels�������е�����ת��Ϊͬһ���������е�����,Ĭ��3������һ����,����dw���
template <
    unsigned int InWidth,   // width of input stream
    unsigned int OutWidth,  // width of output stream
    unsigned int NumInWords,
    unsigned int In_Ch,
    unsigned int In_Bit
>
    void ChannelsToRow(
        hls::stream<ap_uint<InWidth>>& in,
        hls::stream<ap_uint<OutWidth>>& out,
        const unsigned int numReps = 1)
{
    ap_uint<OutWidth> temp[3][In_Ch] = { 0 };
    const unsigned int totalIters = NumInWords * numReps;
    ap_uint<InWidth> ei = 0;
    unsigned int j = 0;
    unsigned int y = 0;
    for (unsigned int t = 0; t < totalIters; t++)
    {
        ei = in.read();
        j++;
        for (unsigned int i = 0; i < In_Ch; i++)
        {
            temp[y][i] = temp[y][i] >> In_Bit;
            temp[y][i]((OutWidth - 1), OutWidth - In_Bit) = ei(In_Bit - 1, 0);
            ei = ei >> In_Bit;
        }
        if (j == 3) {
            y++;
            j = 0;
            if (y == 3)
            {
                for (unsigned int i = 0; i < In_Ch; i++) {
                    for (unsigned int y = 0; y < 3; y++) {
                        out.write(temp[y][i]);
                    }
                }
                y = 0;
            }
        }

    }
}


template <
    unsigned int InWidth,   // width of input stream
    unsigned int OutWidth,  // width of output stream
    unsigned int NumInWords,
    unsigned int In_Ch,
    unsigned int In_Bit,
	unsigned int Out_BIT
>
void ChannelsToRow_simd3(
    hls::stream<ap_uint<InWidth>>& in,
    hls::stream<ap_uint<OutWidth>>& out,
    const unsigned int numReps = 1)
{
    ap_uint<9*Out_BIT> temp[In_Ch] ;
#pragma HLS RESOURCE variable=temp core=RAM_2P_URAM
    const unsigned int totalIters = NumInWords * numReps;
    ap_uint<InWidth> ei = 0;
    total_loop:
    for (unsigned int t = 0; t < totalIters / 9; t++)
    {
    	for (unsigned int n = 0; n < 9; n++){
    		ei = in.read();
    		input_loop:for (unsigned int i = 0; i < In_Ch; i++)
			{
#pragma HLS PIPELINE
				temp[i] = temp[i] >> In_Bit;
				temp[i]((9*Out_BIT - 1), 9*Out_BIT - In_Bit) = ei(In_Bit - 1, 0);
				ei = ei >> In_Bit;
			}
    	}
    	for (unsigned int i = 0; i < In_Ch; i++) {
    		output_loop:for (unsigned int y = 0; y < 3; y++) {
#pragma HLS PIPELINE
				out.write(temp[i](3 * Out_BIT-1,0));
				temp[i] = temp[i]>>(3 * Out_BIT);
			}
		}
    }
}


template <
    unsigned int InWidth,   // width of input stream
    unsigned int OutWidth,  // width of output stream
    unsigned int NumInWords,
    unsigned int In_Ch,
    unsigned int In_Bit,
	unsigned int Out_BIT
>
void ChannelsToRow_simd9(
    hls::stream<ap_uint<InWidth>>& in,
    hls::stream<ap_uint<OutWidth>>& out,
    const unsigned int numReps = 1)
{
    ap_uint<9*Out_BIT> temp[In_Ch] ;
#pragma HLS RESOURCE variable=temp core=RAM_2P_LUTRAM

    const unsigned int totalIters = NumInWords * numReps;
    ap_uint<InWidth> ei = 0;
    total_loop:
    for (unsigned int t = 0; t < totalIters / 9; t++)
    {
    	input_9_loop:for (unsigned int n = 0; n < 9; n++){
    		ei = in.read();
    		input_loop:for (unsigned int i = 0; i < In_Ch; i++)
			{
#pragma HLS PIPELINE
//#pragma HLS PIPELINE rewind
				temp[i] = temp[i] >> In_Bit;
				temp[i]((9*Out_BIT - 1), 9*Out_BIT - In_Bit) = ei(In_Bit - 1, 0);
				ei = ei >> In_Bit;
			}
    	}
    	out_loop:for (unsigned int i = 0; i < In_Ch; i++) {
#pragma HLS PIPELINE
//#pragma HLS PIPELINE rewind
			out.write(temp[i]);
		}
    }
}


/**
    * ������㵥Ԫ ͬʱ����bn_���뼤���
    * �ھ��������������������õ��������ֵ
    * ֻ���� 3x3 �ľ�� K = 3, P = 1 S = 2
    * �������ݿ�� Ϊ IN_STREAM_BIT
    * ������ݿ��Ϊ PE * OUT_BIT
    */
template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_bn_act_s2(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][((IN_CH * 9) / SIMD) * (OUT_CH / PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("samepad_out");
    padding_s2<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);
    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
    SWU_URAM<3, 2, INTER_ROW - 1, INTER_COL - 1, IN_CH, IN_BIT>(padding_out, swu_out, reps);
    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit<IN_CH * 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}

/**
 * ������㵥Ԫ ͬʱ����bn_���뼤���
 * �ھ��������������������õ��������ֵ
 * ֻ���� 1x1 �ľ�� K = 1, P = 1 S = 1
 */
template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv1x1_bn_act(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][((IN_CH) / SIMD) * (OUT_CH / PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH* IN_BIT, SIMD* IN_BIT, OUT_ROW* OUT_COL>(in, adj_out, reps);
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit<IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);

    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}

/**
 * ��ȿɷ��������㵥Ԫ ͬʱ����bn_���뼤���
 * �ھ��������������������õ��������ֵ
 * ֻ���� 3x3 �ľ�� K = 3, P = 1 S = 1
 * �������ݿ�� Ϊ IN_STREAM_BIT
 * ������ݿ��Ϊ PE * OUT_BIT
 */
template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_dw_bn_act_simd3(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][OUT_CH * 3 * 3 / (SIMD * PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM

    padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_LUTRAM

    SWU<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    ChannelsToRow_simd3<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL, IN_CH, IN_BIT, OUT_BIT>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit_dw< 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}

template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_dw_bn_act_simd3_SWU_URAM(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][OUT_CH * 3 * 3 / (SIMD * PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM

    padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_LUTRAM

    SWU_URAM<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    ChannelsToRow_simd3<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL, IN_CH, IN_BIT, OUT_BIT>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit_dw< 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}

/**
 * ��ȿɷ��������㵥Ԫ ͬʱ����bn_���뼤���
 * �ھ��������������������õ��������ֵ
 * ֻ���� 3x3 �ľ�� K = 3, P = 1 S = 2
 * �������ݿ�� Ϊ IN_STREAM_BIT
 * ������ݿ��Ϊ PE * OUT_BIT
 */
template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_dw_bn_act_s2_simd3(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][OUT_CH * 3 * 3 / (SIMD * PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 1;
    const unsigned INTER_COL = IN_COL + 1;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("padding_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM

    padding_s2<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_LUTRAM

    SWU<3, 2, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    ChannelsToRow_simd3<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL, IN_CH, IN_BIT, OUT_BIT>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit_dw<
        3 * 3, OUT_CH,
        IN_BIT, OUT_BIT,
        W_BIT, M_BIT,
        INC_BIT, BIAS_BIT,
        SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}

template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_dw_bn_act_s2_simd3_SWU_URAM(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][OUT_CH * 3 * 3 / (SIMD * PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 1;
    const unsigned INTER_COL = IN_COL + 1;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("padding_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM

    padding_s2<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_LUTRAM

    SWU_URAM<3, 2, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    ChannelsToRow_simd3<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL, IN_CH, IN_BIT, OUT_BIT>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit_dw<
        3 * 3, OUT_CH,
        IN_BIT, OUT_BIT,
        W_BIT, M_BIT,
        INC_BIT, BIAS_BIT,
        SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}


template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_dw_bn_act_simd9(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][OUT_CH * 3 * 3 / (SIMD * PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 2;
    const unsigned INTER_COL = IN_COL + 2;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("samepad_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM
    padding<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_LUTRAM
    SWU_URAM<3, 1, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    ChannelsToRow_simd9<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL, IN_CH, IN_BIT, OUT_BIT>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit_dw< 3 * 3, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, INC_BIT, BIAS_BIT, SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}



/**
 * ��ȿɷ��������㵥Ԫ ͬʱ����bn_���뼤���
 * �ھ��������������������õ��������ֵ
 * ֻ���� 3x3 �ľ�� K = 3, P = 1 S = 2
 * �������ݿ�� Ϊ IN_STREAM_BIT
 * ������ݿ��Ϊ PE * OUT_BIT
 */
template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,
    unsigned INC_BIT,
    unsigned BIAS_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv3x3_dw_bn_act_s2_simd9(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][OUT_CH * 3 * 3 / (SIMD * PE)],
        const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
        const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned INTER_ROW = IN_ROW + 1;
    const unsigned INTER_COL = IN_COL + 1;
    // ��ʱ��Ϊ���� ���ά�Ȳ���
    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2;

    // stream<ap_uint<IN_CH*IN_BIT> > in_adj("in_adj");
    // StreamingDataWidthConverter_Batch<IN_STREAM_BIT, IN_CH*IN_BIT>(in, in_adj, reps);
    // pading
    stream<ap_uint<IN_CH* IN_BIT> > padding_out("padding_out");
#pragma HLS RESOURCE variable=padding_out core=FIFO_LUTRAM
    padding_s2<IN_ROW, IN_COL, IN_CH, IN_BIT, 1>(in, padding_out, reps);

    //����
    stream<ap_uint<IN_CH* IN_BIT> > swu_out("swu_out");
#pragma HLS RESOURCE variable=swu_out core=FIFO_LUTRAM
    SWU_URAM<3, 2, INTER_ROW, INTER_COL, IN_CH, IN_BIT>(padding_out, swu_out, reps);

    // λ�����
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    ChannelsToRow_simd9<IN_CH* IN_BIT, SIMD* IN_BIT, 9 * OUT_ROW * OUT_COL, IN_CH, IN_BIT, OUT_BIT>(swu_out, adj_out, reps);

    // cout << "adj_out size " << adj_out.size() << endl;
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_act_unit_dw<
        3 * 3, OUT_CH,
        IN_BIT, OUT_BIT,
        W_BIT, M_BIT,
        INC_BIT, BIAS_BIT,
        SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, inc, bias, mvau_out, reps);
    // cout << "mvau_out size " << mvau_out.size() << endl;
    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}


template<
    unsigned int ImgDim,
    unsigned int NumChannels,
    typename ActType,

    unsigned int PECount,
    unsigned int ShiftNum,
    typename AccType,
    typename OutType


>
void QuantAvgPool(
    stream<ap_uint<NumChannels* ActType::width> >& in,
    stream<ap_uint<NumChannels* OutType::width> >& out,
    const unsigned int numReps)
{
#pragma HLS DATAFLOW

    stream<ap_uint<PECount* ActType::width> > dwc_out("dwc_out");
    StreamingDataWidthConverter_Batch <NumChannels* ActType::width, PECount* ActType::width, ImgDim* ImgDim>
        (in, dwc_out, numReps);

    stream<ap_uint<PECount* OutType::width> > avg_out("avg_out");
    QuantAvgPool_Batch<ImgDim, NumChannels, ActType, PECount, ShiftNum, AccType, OutType>
        (dwc_out, avg_out, numReps);


    //stream<ap_uint<NumChannels* OutType::width> > out_buffer("out_buffer");
    StreamingDataWidthConverter_Batch <PECount* OutType::width, NumChannels* OutType::width, NumChannels / PECount>
        (avg_out, out, numReps);


}



/**
 * ������㵥Ԫ ͬʱ����bn_���뼤���
 * �ھ��������������������õ��������ֵ
 * ֻ���� 1x1 �ľ�� K = 1, P = 1 S = 1
 */
template <
    unsigned IN_ROW,
    unsigned IN_COL,
    unsigned IN_CH,
    unsigned IN_BIT,

    unsigned OUT_CH,
    unsigned OUT_BIT,		// ����������λ��

    unsigned W_BIT,
    unsigned M_BIT,

    unsigned SIMD,
    unsigned PE,
    unsigned L_SHIFT>
    void conv1x1_fclayer(
        stream<ap_uint<IN_BIT* IN_CH> >& in,
        const ap_uint<SIMD* W_BIT> weights[PE][((IN_CH) / SIMD) * (OUT_CH / PE)],
        stream<ap_uint<OUT_BIT* OUT_CH> >& out,
        const unsigned reps = 1)
{
#pragma HLS DATAFLOW

    const unsigned OUT_ROW = IN_ROW;
    const unsigned OUT_COL = IN_COL;
    stream<ap_uint<SIMD* IN_BIT> > adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH* IN_BIT, SIMD* IN_BIT, OUT_ROW* OUT_COL>(in, adj_out, reps);
    // ������������
    stream<ap_uint<PE* OUT_BIT> > mvau_out("mvau_out");
    matrix_vector_noact_unit<IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, SIMD, PE, L_SHIFT, OUT_ROW* OUT_COL>
        (adj_out, weights, mvau_out, reps);

    StreamingDataWidthConverter_Batch<PE* OUT_BIT, OUT_CH* OUT_BIT, OUT_ROW* OUT_COL* OUT_CH / PE>(mvau_out, out, reps);
}



