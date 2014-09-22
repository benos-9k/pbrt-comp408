#include "bentex.h"

Spectrum BenTexture::Evaluate(const DifferentialGeometry &dg) const {
	
	Point p;
	Vector dpdx, dpdy;
	p = map->Map(dg, &dpdx, &dpdy);

	// the min value of FBm() is less than -1. wtf?

	// sawtooth-ish function (smooth)
	auto thing = [&](float x) {
		return pow(pow(cos(M_PI * x), 2), 0.1 + pow(fmod(fmod(x, 1) + 1, 1), 3) * 30);
	};

	Point p2 = p;
	// make the primary noise less dependent on the 'grain plane'
	p2.x *= 0.08;
	p2.z *= 0.08;
	// introduce a wobble to create uneven grain spacing
	// i dont think this does exactly what i want; oh dear, how sad, never mind.
	p2.y += wobble2 * pow(sin(wobble2_freq * (2.0 * M_PI * p2.y + wobble1 * FBm(wobble1_freq * p, dpdx, dpdy, 1, 2))), 2.0);
	// noise and scale
	float n = FBm(wobble0_freq * p2, dpdx, dpdy, 1.0, 4);
	float m = wobble0 * n;

	// eval points for wood 'waves'
	float x0 = grain0_freq * (p.y + m);
	float x1 = grain1_freq * (p.y + m);

	// interpolation noise
	Point p3 = p;
	p3.x *= noise_stretch;
	p3.z *= noise_stretch;
	float b0 = noise_weight * (FBm(noise_freq * p3, dpdx, dpdy, 1.0, 8) + n);

	// light <-> dark interpolant
	float a0 = (0.6 + b0) * thing(x0) + (0.4 - b0) * thing(x1);
	
	// colour blotchiness noise
	float b2 = blotch_weight * FBm(blotch_freq * p, dpdx, dpdy, 0.5, 8);

	return ((0.6 + b2) * color0 + (0.4 - b2) * color1) * (1.0 - a0) + color2 * a0;
}

Texture<float> * CreateBenFloatTexture(const Transform &tex2world, const TextureParams &tp) {
	return nullptr;
}

BenTexture * CreateBenSpectrumTexture(const Transform &tex2world, const TextureParams &tp) {
	TextureMapping3D *map = new IdentityMapping3D(tex2world);
	return new BenTexture(map, tp);
}