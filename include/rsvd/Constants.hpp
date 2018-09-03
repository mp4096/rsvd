#ifndef RSVD_CONSTANTS_HPP_
#define RSVD_CONSTANTS_HPP_

namespace Rsvd {

/// \brief Enum for conditioners to be used during subspace iterations
///
/// \warning Modified Gram--Schmidt is a self-implemented algorithm, use it with caution. Use
/// #LuConditioner or #QrConditioner if numerical stability and implementation quality is
/// important.
enum SubspaceIterationConditioner {
  NoConditioner,  ///< No conditioner, fastest, can cause numerical problems
  LuConditioner,  ///< Fully pivoted LU decompostion, fast, acceptable numerical
                  ///< stability
  MgsConditioner, ///< Modified Gram--Schmidt orthonormalization, slow,
                  ///< better numerical stability
  QrConditioner,  ///< QR decomposition (Householder reflections), slowest,
                  ///< best numerical stability
};

} // namespace Rsvd

#endif
