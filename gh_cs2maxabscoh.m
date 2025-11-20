function [res, wa, wb, ta, tb] = gh_cs2maxabscoh(csa, csb, csab)
%CS2MAXABSCOH_REGD  Maximize absolute coherence between two sensor spaces.
%
%   [res, wa, wb, ta, tb] = gh_cs2maxabscoh(csa, csb, csab)
%
%   Constructs spatial filters (weights) in two sensor spaces (A and B) such
%   that the coherence between the corresponding "virtual sensors" is maximized.
%   The virtual sensors are linear combinations of the channels:
%       vA = wa' * signalA
%       vB = wb' * signalB
%
%   INPUTS
%   -------
%   csa  : (N x N) real/complex cross-spectrum in space A (Hermitian).
%          Only the real part is used.
%   csb  : (M x M) real/complex cross-spectrum in space B (Hermitian).
%          Only the real part is used.
%   csab : (N x M) complex cross-spectrum between spaces A and B.
%
%   OUTPUTS
%   --------
%   res : complex coherence (maximum absolute coherence value).
%   wa  : weights (filter) for space A (in sensor space).
%   wb  : weights (filter) for space B (in sensor space).
%   ta  : topography (pattern) for space A   = csa * wa, normalized.
%   tb  : topography (pattern) for space B   = csb * wb, normalized.
%
%   Method:
%       - Regularizes and reduces dimensionality of csa, csb using SVD retaining
%         99% of variance.
%       - Computes reduced cross-spectrum.
%       - Performs a 1-D optimization over a phase parameter φ to maximize
%         |coherence|.
%       - Computes spatial filters and patterns, then maps back to original space.
%
%   -------------------------------------------------------------------------


%% --- Dimensionality reduction for both spaces ---
[UA, csrA, keepA] = reduce_dim(csa);
[UB, csrB, keepB] = reduce_dim(csb);

%% --- Reduced cross-spectrum between A and B ---
csrAB = UA' * csab * UB;

%% --- Precompute inverses & whitening ---
cbb_inv      = inv(csrB);
caa_inv_sqrt = sqrtm(inv(csrA));

%% --- Coarse φ scan ---
nIterCoarse = 5;
resCoarse   = zeros(nIterCoarse, 1);

for it = 1:nIterCoarse
    phi = (it-1) / nIterCoarse * pi;
    resCoarse(it) = cohmax(phi, csrAB, cbb_inv, caa_inv_sqrt);
end

[~, idxMax] = max(resCoarse);
phi = (idxMax-1) / nIterCoarse * pi;

%% --- Fine optimization of φ ---
nIterFine = 10;
dphi = 1e-6;
akont = 1;

for it = 1:nIterFine
    f0 = cohmax(phi, csrAB, cbb_inv, caa_inv_sqrt);
    fp = cohmax(phi + dphi, csrAB, cbb_inv, caa_inv_sqrt);
    fm = cohmax(phi - dphi, csrAB, cbb_inv, caa_inv_sqrt);

    fprime = (fp - fm) / (2*dphi);
    fpp    = (fp + fm - 2*f0) / (dphi^2);

    deltaPhi = -fprime / (fpp - akont);
    phiNew    = phi + deltaPhi;
    phiNew    = mod(phiNew + pi/2, pi) - pi/2;

    fNew = cohmax(phiNew, csrAB, cbb_inv, caa_inv_sqrt);

    if fNew > f0
        akont = akont / 2;
        phi   = phiNew;
    else
        akont = akont * 2;
    end
end

%% --- Compute spatial filters in reduced space ---
[res_val, wa_red, wb_red, ta_red, tb_red] = ...
    cohmax_withdirs(phi, csrAB, csrA, csrB);

% Final coherence magnitude
res = abs(res_val * exp(-1i * phi));

%% --- Map to original sensor space ---
wa = UA * wa_red;
wb = UB * wb_red;

ta = UA * ta_red;
tb = UB * tb_red;

% Normalize topographies
ta = ta / norm(ta);
tb = tb / norm(tb);

end
%% ========================================================================
%                       Helper Functions
%  ========================================================================

function resloc = cohmax(phi, csrAB, cbb_inv, caa_inv_sqrt)
%COHMAX  Compute coherence magnitude for a given phase φ (reduced space).
%
%   This implements the core expression under a phase rotation:
%       C_AB(φ) = real( exp(iφ) * C_AB )
%
%   Returns the largest singular value sqrt(λ_max).

    cab = real(exp(1i*phi) * csrAB);           % rotated cross-spectrum
    X   = cab * cbb_inv * cab';                % numerator term
    Y   = caa_inv_sqrt * X * caa_inv_sqrt;     % whitened space
    s   = svd(Y);                              
    resloc = sqrt(s(1));                       % maximal coherence
end



function [resloc, wa, wb, ta, tb] = cohmax_withdirs(phi, csrAB, csrA, csrB)
%COHMAX_WITHDIRS  Compute maximizing coherence AND return directions.
%
%   Same as cohmax(), but returns:
%       wa, wb = spatial filters
%       ta, tb = topographies

    cab = real(exp(1i*phi) * csrAB);

    caa_inv      = inv(csrA);
    cbb_inv      = inv(csrB);
    caa_inv_sqrt = sqrtm(caa_inv);
    cbb_inv_sqrt = sqrtm(cbb_inv);

    % whitened cross-spectrum
    D = caa_inv_sqrt * cab * cbb_inv_sqrt;

    % singular vectors give directions achieving max coherence
    [U, S, ~] = svd(D * D');    
    wA_proj = U(:,1);
    resloc  = sqrt(S(1,1));

    % unwhiten weights
    wa = caa_inv_sqrt * wA_proj;
    wa = wa / norm(wa);

    wB_proj = D' * wA_proj;
    wb = cbb_inv_sqrt * wB_proj;
    wb = wb / norm(wb);

    % compute patterns (topographies)
    ta = csrA * wa;
    tb = csrB * wb;
end

function [Ured, Cred, keepIdx] = reduce_dim(C)
%REDUCE_DIM  Regularize, SVD-reduce, and return reduced covariance matrix.
%
%   INPUT:
%       C : (N x N) real/complex covariance/cross-spectral matrix
%
%   OUTPUT:
%       Ured    : (N x K) basis vectors for reduced space
%       Cred    : (K x K) reduced covariance
%       keepIdx : indices of retained singular values (≥ 99% variance)

    % use only real part
    C = real(C);
    N = size(C, 1);

    % small regularization
    Creg = C + eye(N) * mean(diag(C)) * 1e-10;

    % SVD
    [U, S, ~] = svd(Creg);
    s = diag(S);
    s_norm = s / sum(s);

    % keep ≥ 99% variance
    [~, idx] = sort(s_norm, 'descend');
    keepIdx = idx(cumsum(s_norm(idx)) <= 0.99);
    if isempty(keepIdx)
        keepIdx = idx(1);
    end

    % reduced basis and covariance
    Ured = U(:, keepIdx);
    Cred = Ured' * C * Ured;
end
