

//! ----------------------------------------------------------------------------- ARMADILLO HELPERS -----------------------------------------------------------------------------

//<! calculate commutator of two input matrix types, which have overloaded * operator
inline std::string matrix_size(u64 dim){
	 if(dim < 1e3)
	 	return std::to_string(dim) + " bytes";
	 else if(dim < 1e6)
	 	return to_string_prec(dim / 1e3, 2) + " kB";
	 else if(dim < 1e9)
	 	return to_string_prec(dim / 1e6, 2) + " MB";
	 else if(dim < 1e12)
	 	return to_string_prec(dim / 1e9, 2) + " GB";
	else 
	 	return to_string_prec(dim / 1e12, 2) + " TB";
}


template <typename matrix>
matrix commutator(const matrix& A, const matrix& B)
	{ return A * B - B * A; }

//! -------------------------------------------------------- cast non-cpx to cpx types
template <typename _ty>
arma::Col<std::complex<_ty>> cpx_real_vec(const arma::Col<_ty>& input){ 
	size_t size = input.size();
	return arma::Col<std::complex<_ty>>(input, arma::Col<_ty>(size, arma::fill::zeros));
}
template <typename _ty>
arma::Col<std::complex<_ty>> cpx_imag_vec(const arma::Col<_ty>& input) {
	size_t size = input.size();
	return arma::Col<std::complex<_ty>>(arma::Col<_ty>(size, arma::fill::zeros), input);
}
template <typename _ty>
arma::Col<std::complex<_ty>> cpx_real_vec(const arma::subview_col<_ty>& input) {
	size_t size = input.n_elem;
	return arma::Col<std::complex<_ty>>(input, arma::Col<_ty>(size, arma::fill::zeros));
}
template <typename _ty>
arma::Col<std::complex<_ty>> cpx_imag_vec(const arma::subview_col<_ty>& input) {
	size_t size = input.n_elem;
	return arma::Col<std::complex<_ty>>(arma::Col<_ty>(size, arma::fill::zeros), input);
}


template <typename _type>
inline
arma::cx_vec cast_cx_vec(const arma::Col<_type>& state);

template <>
inline arma::cx_vec cast_cx_vec(const arma::vec& state)
	{ return cpx_real_vec(state); }
template <>
inline arma::cx_vec cast_cx_vec(const arma::cx_vec& state)
	{ return state; }
//! -------------------------------------------------------- dot product for different input types (cpx and non-cpx)

 template <typename _ty, 
	 template <typename> class _COLVEC1,
	 template <typename> class _COLVEC2 
 >
_ty dot_prod(const _COLVEC1<_ty>& left, const _COLVEC2<_ty>& right)
{ 
	 static_assert(traits::is_any_of_v<_COLVEC1<_ty>, arma::Col<_ty>, arma::subview_col<_ty>>
		 && traits::is_any_of_v<_COLVEC2<_ty>, arma::Col<_ty>, arma::subview_col<_ty>>,
		 "Dot product only valid for arma::Col and arma::subview classes");
	return arma::cdot(left, right); 
}																			
																											
template <typename _ty,
	template <typename> class _COLVEC1,
	template <typename> class _COLVEC2
>
std::complex<_ty> dot_prod(const _COLVEC1<_ty>& left, const _COLVEC2<std::complex<_ty>>& right)
{
	static_assert(traits::is_any_of_v<_COLVEC1<_ty>, arma::Col<_ty>, arma::subview_col<_ty>>
		&& traits::is_any_of_v<_COLVEC2<std::complex<_ty>>, arma::Col<std::complex<_ty>>, arma::subview_col<std::complex<_ty>>>,
		"Dot product only valid for arma::Col and arma::subview classes");
	return arma::cdot(cpx_real_vec(left), right);
}
																											
template <typename _ty,
	template <typename> class _COLVEC1,
	template <typename> class _COLVEC2
>
std::complex<_ty> dot_prod(const _COLVEC1<std::complex<_ty>> & left, const _COLVEC2<_ty> & right)
{
	static_assert(traits::is_any_of_v<_COLVEC1<std::complex<_ty>>, arma::Col<std::complex<_ty>>, arma::subview_col<std::complex<_ty>>>
		&& traits::is_any_of_v<_COLVEC2<_ty>, arma::Col<_ty>, arma::subview_col<_ty>>,
		"Dot product only valid for arma::Col and arma::subview classes"); 
	return arma::cdot(left, cpx_real_vec(right));
}															
											
template <typename _ty>
inline arma::Col<_ty> exctract_vector_between_values(
	const arma::Col<_ty>& input_vec,	//<! input vector to exctract data from (assumed sorted)
	_ty start, 							//<! first value of new vector (if lower than lowest in input_vec than taking from beggining)
	_ty end								//<! last element to copy data
) {
	arma::Col<_ty> output;
	for (auto& it : input_vec) {
		if (it >= start && it <= end) {
			int size = output.size();
			output.resize(size + 1);
			output(size) = it;
		}
	}
	return output;
}
inline arma::vec exctract_vector(
	const arma::vec& input_vec,	//<! input vector to exctract data from (assumed sorted)
	u64 start, 	//<! first value of new vector (if lower than lowest in input_vec than taking from beggining)
	u64 end		//<! last element to copy data
) {
	arma::vec output(end - start);
#pragma omp parallel for
	for (int k = start; k < end; k++) 
		output(k - start) = input_vec(k);
	return output;
}

template <typename _type>
inline
arma::sp_cx_mat cast_cx_sparse(const arma::SpMat<_type>& mat);

template <>
inline arma::sp_cx_mat cast_cx_sparse(const arma::sp_mat& mat)
{
	arma::sp_cx_mat ret(mat.n_rows, mat.n_cols);
	ret.set_real(mat);
	return ret;
}
template <>
inline arma::sp_cx_mat cast_cx_sparse(const arma::sp_cx_mat& mat)
	{ return mat; }
//general_dot_prod(arma::Col,			arma::Col		 );
//general_dot_prod(arma::subview_col, arma::Col		 );
//general_dot_prod(arma::Col,			arma::subview_col);
//general_dot_prod(arma::subview_col, arma::subview_col);
