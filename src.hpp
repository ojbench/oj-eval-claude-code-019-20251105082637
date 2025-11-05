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

    // For each key-value pair from 0 to i
    for (size_t j = 0; j <= i; ++j) {
      // Move key and value to SRAM
      gpu_sim.MoveMatrixToSharedMem(keys[j]);
      gpu_sim.MoveMatrixToSharedMem(values[j]);

      // Compute Q * K^T for current key
      // Q shape: [i+1, 512], K shape: [1, 512] -> Q * K^T shape: [i+1, 1]
      // First transpose K to get K^T shape: [512, 1]
      Matrix* k_transposed = matrix_memory_allocator.Allocate("k_transposed_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.Transpose(keys[j], kInSharedMemory);
      gpu_sim.Copy(keys[j], k_transposed, kInSharedMemory);

      Matrix* qk_result = matrix_memory_allocator.Allocate("qk_result_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatMul(current_query, k_transposed, qk_result);

      // Compute exp(Q * K^T) for softmax
      Matrix* exp_qk = matrix_memory_allocator.Allocate("exp_qk_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatExp(qk_result, exp_qk);

      // Sum over rows for softmax denominator
      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.Sum(exp_qk, row_sum);

      // Compute softmax: exp(QK^T) / sum(exp(QK^T))
      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatDiv(exp_qk, row_sum, softmax_row);

      // Compute softmax * V for current key-value pair
      // softmax shape: [i+1, 1], V shape: [1, 512] -> result shape: [i+1, 512]
      Matrix* attention_row = matrix_memory_allocator.Allocate("attention_row_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatMul(softmax_row, values[j], attention_row);

      // Add to running total
      Matrix* new_attention = matrix_memory_allocator.Allocate("new_attention_" + std::to_string(i) + "_" + std::to_string(j));
      gpu_sim.MatAdd(attention_result, attention_row, new_attention);

      // Update attention_result
      gpu_sim.ReleaseMatrix(attention_result);
      attention_result = new_attention;

      // Clean up intermediate matrices for this key-value pair
      gpu_sim.ReleaseMatrix(k_transposed);
      gpu_sim.ReleaseMatrix(qk_result);
      gpu_sim.ReleaseMatrix(exp_qk);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(softmax_row);
      gpu_sim.ReleaseMatrix(attention_row);
    }

    // Move final result to HBM
    gpu_sim.MoveMatrixToGpuHbm(attention_result);

    // Run simulator to execute all queued operations
    gpu_sim.Run(false, &matrix_memory_allocator);

    // Commit the answer
    rater.CommitAnswer(*attention_result);

    // Clean up remaining matrices
    gpu_sim.ReleaseMatrix(zero_matrix);
    gpu_sim.ReleaseMatrix(attention_result);

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