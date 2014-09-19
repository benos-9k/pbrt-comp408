#include "bentex.h"

Spectrum BenTexture::Evaluate(const DifferentialGeometry &dg) const {
	
	Point p;
	Vector dpdx, dpdy;
	p = map->Map(dg, &dpdx, &dpdy);

	// plot(x, (cos(pi * x) .^ 2) .^ (0.1 + mod(x, 1) .^ 3 * 15))

	auto thing = [&](float x) {
		return pow(pow(cos(M_PI * x), 2), 0.1 + pow(fmod(fmod(x, 1) + 1, 1), 3) * 30);
	};

	Point p2 = p;
	p2.x *= 0.1;
	p2.z *= 0.1;

	float x0 = 40 * (p.y + 0.06 * FBm(p2, dpdx, dpdy, 1.0, 4));
	float x1 = 200 * (p.y + 0.06 * FBm(p2, dpdx, dpdy, 1.0, 4));
	float b = 0.2 * FBm(p * 100, dpdx, dpdy, 1.0, 8);
	float a0 = (0.6 + b) * thing(x0) + (0.3 - b) * thing(x1);
	
	float x5 = 0.05 * Distance(p, Point()) * (3 + FBm(p, dpdx, dpdy, 0.5, 8));
	
	Point pp;
	p.x *= 10;
	p.z *= 10;
	//a += 0.5 * FBm(pp, dpdx, dpdy, 0.5, 7);

	float c0[3] { 192.0 / 255.0, 134.0 / 255.0, 84.0 / 255.0 };
	float c1[3] { 105.0 / 255.0, 63.0 / 255.0, 28.0 / 255.0 };
	float c2[3] { 211.0 / 255.0, 108.0 / 255.0, 22.0 / 255.0 };

	Spectrum s0 = Spectrum::FromRGB(c0);
	Spectrum s1 = Spectrum::FromRGB(c1);
	Spectrum s2 = Spectrum::FromRGB(c2);
	

	return 0.8 * (s0 * (1.0 - a0) + s1 * a0) + 0.2 * FBm(p, dpdx, dpdy, 0.5, 8) * s2;
}

Texture<float> * CreateBenFloatTexture(const Transform &tex2world, const TextureParams &tp) {
	return nullptr;
}

BenTexture * CreateBenSpectrumTexture(const Transform &tex2world, const TextureParams &tp) {
	TextureMapping3D *map = new IdentityMapping3D(tex2world);
	return new BenTexture(map);
}