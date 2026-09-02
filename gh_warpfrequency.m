function x_warped = gh_warpfrequency(x, FrBase, factor, fs)
% Copyright (C) 2026 Carmen Vidaurre

% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.

% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details

%WARP_FREQUENCY  Frequency-warp a narrow-band component of a multichannel signal.
%
%   x_warped = gh_WARPFREQUENCY(x, FrBase, factor, fs)
%
%   This function extracts a narrow band centered around a base frequency
%   (FrBase), computes its analytic representation, multiplies the
%   instantaneous phase by a given factor, and reconstructs a real-valued
%   warped signal. This effectively shifts/warps the component from
%   FrBase → FrBase * factor (approximately).
%
%   INPUTS:
%       x       - Time-domain signal, size [T × Nchannels]
%       FrBase  - Base frequency of interest (Hz), e.g., 3 Hz
%       factor  - Frequency multiplication factor, e.g., 2 → from 3 Hz to 6 Hz
%       fs      - Sampling frequency (Hz)
%
%   OUTPUT:
%       x_warped - Frequency-warped signal, same size as x
%
%   EXAMPLE:
%       % Warp a 3 Hz oscillation to 6 Hz:
%       y = warp_frequency(x, 3, 2, fs);
%
%   NOTE:
%       The method uses:
%         - narrow band-pass filtering
%         - analytic signal via Hilbert transform
%         - instantaneous phase multiplication
%
%   Author: Carmen Vidaurre
% Please cite: Vidaurre, C., Eguinoa, R., Maudrich, T. et al. Canonical coherence 
% for the estimation of within- and cross-frequency cortico-kinematic interactions. 
% Sci Rep 16, 15182 (2026). 
% https://doi.org/10.1038/s41598-026-49471-6
% -------------------------------------------------------------------------

%% -----------------------
%  Input validation
% ------------------------
if nargin < 4
    error('warp_frequency requires inputs: x, FrBase, factor, fs.');
end
if FrBase <= 0
    error('FrBase must be positive.');
end
if factor <= 0
    error('factor must be positive.');
end
if fs <= 0
    error('Sampling frequency fs must be positive.');
end

% Ensure column-oriented time dimension
if size(x,1) < size(x,2)
    warning('Input x appears transposed. Expected [T × channels].');
end


%% -----------------------
%  Design narrow band-pass filter
% ------------------------
% Bandwidth: ±1 Hz around FrBase
f_low  = max(FrBase - 1, 0.1);       % prevent 0 Hz edge
f_high = FrBase + 1;

% Normalize to Nyquist frequency
Wn = [f_low  f_high] / (fs/2);

% 4th-order Butterworth BPF
[b, a] = butter(2, Wn);


%% -----------------------
%  Apply zero-phase band-pass filter
% ------------------------
% filtfilt ensures zero-phase distortion
x_filt = filtfilt(b, a, x);


%% -----------------------
%  Compute analytic signal using Hilbert transform
% ------------------------
x_analytic = hilbert(x_filt);


%% -----------------------
%  Frequency warping
% ------------------------
% Warp = keep amplitude envelope |analytic|
%        but multiply instantaneous phase by `factor`
%
% analytic = A * exp(i * φ)
% warped   = A * exp(i * (factor * φ))

A = abs(x_analytic); % this is the original envelope
phi = angle(x_analytic);

x_warped = real( A .* exp(1i * (factor .* phi)) );


end
