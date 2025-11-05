#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();

    // Allocate matrices for intermediate results
    Matrix* attention_result = matrix_memory_allocator.Allocate("attention_result_" + std::to_string(i));

    // Move query to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Initialize attention_result as zeros with same shape as query
    Matrix* zero_matrix = matrix_memory_allocator.Allocate("zero_matrix_" + std::to_string(i));
    gpu_sim.Copy(current_query, zero_matrix, kInSharedMemory);
    zero_matrix->Zero();
    gpu_sim.Copy(zero_matrix, attention_result, kInSharedMemory);

    // For round i, we need to use keys[0] to keys[i] and values[0] to values[i]
    // The attention computation is: Softmax(Q * K^T) * V
    // But we need to compute this for all keys at once, not separately

    // First, concatenate all keys to form K matrix
    Matrix* K_concat = matrix_memory_allocator.Allocate("K_concat_" + std::to_string(i));
    gpu_sim.Copy(keys[0], K_concat, kInSharedMemory);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      Matrix* new_K_concat = matrix_memory_allocator.Allocate("new_K_concat_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.Concat(K_concat, keys[j], new_K_concat, 0, kInSharedMemory); // Concatenate vertically
      gpu_sim.ReleaseMatrix(K_concat);
      K_concat = new_K_concat;
    }

    // Now K_concat shape: [i+1, 512]
    // Compute Q * K^T
    Matrix* K_transposed = matrix_memory_allocator.Allocate("K_transposed_" + std::to_string(i));
    gpu_sim.Transpose(K_concat, kInSharedMemory);
    gpu_sim.Copy(K_concat, K_transposed, kInSharedMemory);

    Matrix* qk_result = matrix_memory_allocator.Allocate("qk_result_" + std::to_string(i));
    gpu_sim.MatMul(current_query, K_transposed, qk_result); // Q shape: [i+1, 512], K^T shape: [512, i+1] -> result: [i+1, i+1]

    // Compute softmax along rows
    Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk_" + std::to_string(i));
    gpu_sim.MatExp(qk_result, exp_qk);

    // Sum over rows for softmax denominator
    Matrix* row_sums = matrix_memory_allocator.Allocate("row_sums_" + std::to_string(i));
    gpu_sim.Sum(exp_qk, row_sums);

    // Compute softmax: exp(QK^T) / sum(exp(QK^T)) for each row
    Matrix* softmax_result = matrix_memory_allocator.Allocate("softmax_result_" + std::to_string(i));
    gpu_sim.MatDiv(exp_qk, row_sums, softmax_result);

    // Concatenate all values to form V matrix
    Matrix* V_concat = matrix_memory_allocator.Allocate("V_concat_" + std::to_string(i));
    gpu_sim.Copy(values[0], V_concat, kInSharedMemory);

    for (size_t j = 1; j <= i; ++j) {
      gpu_sim.MoveMatrixToSharedMem(values[j]);
      Matrix* new_V_concat = matrix_memory_allocator.Allocate("new_V_concat_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.Concat(V_concat, values[j], new_V_concat, 0, kInSharedMemory); // Concatenate vertically
      gpu_sim.ReleaseMatrix(V_concat);
      V_concat = new_V_concat;
    }

    // Now V_concat shape: [i+1, 512]
    // Compute softmax * V
    gpu_sim.MatMul(softmax_result, V_concat, attention_result); // softmax shape: [i+1, i+1], V shape: [i+1, 512] -> result: [i+1, 512]

    // Move final result to HBM
    gpu_sim.MoveMatrixToGpuHbm(attention_result);

    // Run simulator to execute all queued operations
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*attention_result);

    // Clean up matrices
    gpu_sim.ReleaseMatrix(zero_matrix);
    gpu_sim.ReleaseMatrix(attention_result);
    gpu_sim.ReleaseMatrix(K_concat);
    gpu_sim.ReleaseMatrix(K_transposed);
    gpu_sim.ReleaseMatrix(qk_result);
    gpu_sim.ReleaseMatrix(exp_qk);
    gpu_sim.ReleaseMatrix(row_sums);
    gpu_sim.ReleaseMatrix(softmax_result);
    gpu_sim.ReleaseMatrix(V_concat);

    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu