#pragma once

/**
 * @file test_template_generalization.hpp
 * @brief Simple test cases to verify that the generic_operator template generalization works correctly
 *        with both the default u64 state type and custom state types.
 */

#include "operator_base.h"

namespace QOps {
    // Test 1: Verify that generic_operator<cpx> uses u64 as default state_ty
    // This should maintain backward compatibility with existing code
    inline void test_default_u64() {
        // This is how existing code uses generic_operator - eigenvalue type only
        // generic_operator<cpx> should be equivalent to generic_operator<cpx, u64>
        auto lambda_u64 = [](u64 state, int site) -> std::pair<u64, cpx> {
            return {state, cpx(1.0, 0.0)};
        };
        
        // Create operator with default u64 state type
        std::function<std::pair<u64, cpx>(u64, int)> kernel_func = lambda_u64;
        auto op = generic_operator<cpx>(10, kernel_func, cpx(1.0));
        
        // Verify we can call the operator
        auto [result_state, result_val] = op(5, 0);
        // result_state should be u64
        static_assert(std::is_same_v<decltype(result_state), u64>, "Default state type should be u64");
    }
    
    // Test 2: Verify that we can use custom state types if desired
    // This shows the flexibility of the generalized template
    inline void test_custom_state_type() {
        // Example with uint32_t as state type
        using state_type = uint32_t;
        
        auto lambda = [](state_type state, int site) -> std::pair<state_type, double> {
            return {state, 1.0};
        };
        
        // Create operator with custom state type
        std::function<std::pair<state_type, double>(state_type, int)> kernel_func = lambda;
        auto op = generic_operator<double, state_type>(10, kernel_func, 1.0);
        
        // Verify we can call the operator
        auto [result_state, result_val] = op(state_type(5), 0);
        // result_state should be uint32_t
        static_assert(std::is_same_v<decltype(result_state), state_type>, "Custom state type should work");
    }
    
    // Test 3: Verify backward compatibility - existing code patterns still work
    inline void test_backward_compatibility() {
        // Common pattern from user_interface_dis_aux.hpp
        using element_type = cpx;
        
        auto kernel = [](u64 state, int site) -> std::pair<u64, element_type> {
            return {state ^ (1ULL << site), element_type(1.0)};
        };
        
        std::function<std::pair<u64, element_type>(u64, int)> kernel_func = kernel;
        std::vector<generic_operator<element_type>> permutation_op;
        
        auto op = generic_operator<element_type>(10, kernel_func, element_type(1.0));
        permutation_op.push_back(op);
        
        // This should compile without any changes to existing code
    }
}
