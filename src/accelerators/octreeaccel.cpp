
#include "octreeaccel.h"

// SSE ftw, just cause i can
#include <xmmintrin.h>

#include <cstring>
#include <iostream>
#include <unordered_map>

#define BEN_OCTREE_MAX_LEAF_VALUES 16

namespace {

	struct out_of_bounds { };

	// sse vector
	class float3 {
	private:
		__m128 m_data;

	public:
		inline float3(__m128 data_) : m_data(data_) { }

		inline float3() : m_data(_mm_setzero_ps()) { }

		inline float3(float v) : m_data(_mm_set1_ps(v)) { }

		inline float3(float x, float y, float z) : m_data(_mm_set_ps(0, z, y, x)) { }

		inline float3(const Point &p) : m_data(_mm_set_ps(0, p.z, p.y, p.x)) { }

		inline float3(const Vector &v) : m_data(_mm_set_ps(0, v.z, v.y, v.x)) { }

		inline operator Point() {
			return Point(x(), y(), z());
		}

		inline operator Vector() {
			return Vector(x(), y(), z());
		}

		inline __m128 data() {
			return m_data;
		}

		inline float x() const {
			float r;
			_mm_store_ss(&r, m_data);
			return r;
		}

		inline float y() const {
			float r;
			_mm_store_ss(&r, _mm_shuffle_ps(m_data, m_data, _MM_SHUFFLE(0, 0, 0, 1)));
			return r;
		}

		inline float z() const {
			float r;
			_mm_store_ss(&r, _mm_shuffle_ps(m_data, m_data, _MM_SHUFFLE(0, 0, 0, 2)));
			return r;
		}

		inline float3 operator-() const {
			return float3(_mm_sub_ps(_mm_setzero_ps(), m_data));
		}

		inline float3 & operator+=(const float3 &rhs) {
			m_data = _mm_add_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator+=(float rhs) {
			m_data = _mm_add_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator+(const float3 &rhs) const {
			return float3(*this) += rhs;
		}

		inline float3 operator+(float rhs) const {
			return float3(*this) + rhs;
		}

		inline friend float3 operator+(float lhs, const float3 &rhs) {
			return float3(rhs) + lhs;
		}

		inline float3 & operator-=(const float3 &rhs) {
			m_data = _mm_sub_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator-=(float rhs) {
			m_data = _mm_sub_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator-(const float3 &rhs) const {
			return float3(*this) -= rhs;
		}

		inline float3 operator-(float rhs) const {
			return float3(*this) - rhs;
		}

		inline friend float3 operator-(float lhs, const float3 &rhs) {
			return float3(_mm_set1_ps(lhs)) -= rhs;
		}

		inline float3 & operator*=(const float3 &rhs) {
			m_data = _mm_mul_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator*=(float rhs) {
			m_data = _mm_mul_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator*(const float3 &rhs) const {
			return float3(*this) *= rhs;
		}

		inline float3 operator*(float rhs) const {
			return float3(*this) * rhs;
		}

		inline friend float3 operator*(float lhs, const float3 &rhs) {
			return float3(rhs) * lhs;
		}

		inline float3 & operator/=(const float3 &rhs) {
			m_data = _mm_div_ps(m_data, rhs.m_data);
			return *this;
		}

		inline float3 & operator/=(float rhs) {
			m_data = _mm_div_ps(m_data, _mm_set1_ps(rhs));
			return *this;
		}

		inline float3 operator/(const float3 &rhs) const {
			return float3(*this) /= rhs;
		}

		inline float3 operator/(float rhs) const {
			return float3(*this) /= rhs;
		}

		inline friend float3 operator/(float lhs, const float3 &rhs) {
			return float3(_mm_set1_ps(lhs)) /= rhs;
		}

		inline float3 operator<(const float3 &rhs) const {
			return float3(_mm_cmplt_ps(m_data, rhs.m_data));
		}

		inline float3 operator<=(const float3 &rhs) const {
			return float3(_mm_cmple_ps(m_data, rhs.m_data));
		}

		inline float3 operator>(const float3 &rhs) const {
			return float3(_mm_cmpgt_ps(m_data, rhs.m_data));
		}

		inline float3 operator>=(const float3 &rhs) const {
			return float3(_mm_cmpge_ps(m_data, rhs.m_data));
		}

		inline float3 operator==(const float3 &rhs) const {
			return float3(_mm_cmpeq_ps(m_data, rhs.m_data));
		}

		inline float3 operator!=(const float3 &rhs) const {
			return float3(_mm_cmpneq_ps(m_data, rhs.m_data));
		}

		inline friend std::ostream & operator<<(std::ostream &out, const float3 &v) {
			out << "(" << v.x() << ", " << v.y() << ", " << v.z() << ")";
			return out;
		}

		inline static float3 abs(const float3 &x) {
			return float3(_mm_andnot_ps(_mm_set1_ps(-0.f), x.m_data));
		}

		inline static float3 max(const float3 &x, const float3 &y) {
			return float3(_mm_max_ps(x.m_data, y.m_data));
		}

		inline static float3 min(const float3 &x, const float3 &y) {
			return float3(_mm_min_ps(x.m_data, y.m_data));
		}

		inline static bool all(const float3 &x) {
			__m128 r1 = x.m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_and_ps(r3, _mm_and_ps(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r != 0.f;
		}

		inline static bool any(const float3 &x) {
			__m128 r1 = x.m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_or_ps(r3, _mm_or_ps(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r != 0.f;
		}

	};

	// axis-aligned bounding box
	class aabb {
	private:
		float3 m_center;
		float3 m_halfsize;

	public:
		inline aabb() { }

		inline aabb(const float3 &center_) : m_center(center_) { }

		inline aabb(const float3 &center_, const float3 &halfsize_) : m_center(center_), m_halfsize(float3::abs(halfsize_)) { }

		inline float3 center() const {
			return m_center;
		}

		inline float3 halfsize() const {
			return m_halfsize;
		}

		inline float3 min() const {
			return m_center - m_halfsize;
		}

		inline float3 max() const {
			return m_center + m_halfsize;
		}

		inline bool contains(const float3 &p) const {
			return float3::all(float3::abs(p - m_center) <= m_halfsize);
		}

		inline bool contains(const aabb &a) const {
			return float3::all(float3::abs(a.m_center - m_center) <= (m_halfsize - a.m_halfsize));
		}

		inline bool contains_partial(const aabb &a) const {
			return float3::any(float3::abs(a.m_center - m_center) <= (m_halfsize - a.m_halfsize));
		}

		inline bool intersects(const aabb &a) const {
			return float3::all(float3::abs(a.m_center - m_center) <= (m_halfsize + a.m_halfsize));
		}

		inline friend std::ostream & operator<<(std::ostream &out, const aabb &a) {
			out << "aabb[" << a.min() << " <= x <= " << a.max() << "]";
			return out;
		}

		inline static aabb fromPoints(const float3 &p0, const float3 &p1) {
			float3 minv = float3::min(p0, p1);
			float3 maxv = float3::max(p0, p1);
			float3 hs = 0.5 * (maxv - minv);
			return aabb(minv + hs, hs);
		}

		inline static aabb fromBBox(const BBox &b) {
			return fromPoints(b.pMin, b.pMax);
		}

	};

}

class OctreeAccel::Node {
public:
	using value_t = Reference<Primitive>;

	struct hash {
		inline size_t operator()(const Reference<Primitive> &r) const {
			return size_t(r.GetPtr());
		}
	};

	using map_t = std::unordered_map<Reference<Primitive>, aabb, hash>;

private:
	aabb m_bound;
	Node *m_children[8];
	map_t m_values;
	size_t m_count = 0;
	bool m_isleaf = true;

	inline unsigned childID(const float3 &p) {
		return 0x7 & _mm_movemask_ps((p >= m_bound.center()).data());
	}
			
	inline void unleafify();

public:
	inline Node(const aabb &a_) : m_bound(a_) {
		// clear child pointers
		std::memset(m_children, 0, 8 * sizeof(Node *));
	}

	Node(const Node &) = delete;
	Node & operator=(const Node &) = delete;

	inline aabb bound() {
		return m_bound;
	}

	inline size_t count() {
		return m_count;
	}

	// move-insert a value, using a specified bounding box
	inline bool insert(const value_t &t, const aabb &a) {
		if (!m_bound.contains_partial(a)) throw out_of_bounds();
		if (m_isleaf && m_count < BEN_OCTREE_MAX_LEAF_VALUES) {
			if (!m_values.insert(std::make_pair(t, a)).second) return false;
		} else {
			// not a leaf or should not be
			unleafify();
			unsigned cid_min = childID(a.min());
			unsigned cid_max = childID(a.max());
			if (cid_min != cid_max) {
				// element spans multiple child nodes, insert into this one
				// TODO
				if (!m_values.insert(std::make_pair(t, a)).second) return false;
			} else {
				// element contained in one child node - create if necessary then insert
				Node *child = m_children[cid_min];
				if (!child) {
					// positive / negative halfsizes
					__m128 h = m_bound.halfsize().data();
					__m128 g = _mm_sub_ps(_mm_setzero_ps(), h);

					// convert int bitmask to (opposite) sse mask
					__m128i m = _mm_set1_epi32(cid_min);
					m = _mm_and_si128(m, _mm_set_epi32(0, 4, 2, 1));
					m = _mm_cmpeq_epi32(_mm_setzero_si128(), m);
					__m128 n = _mm_castsi128_ps(m);

					// vector to a corner of the current node's aabb
					float3 vr(_mm_or_ps(_mm_and_ps(n, g), _mm_andnot_ps(n, h)));
					const float3 c = m_bound.center();

					child = new Node(aabb::fromPoints(c, c + vr));
					m_children[cid_min] = child;
				}
				try {
					if (!child->insert(t, a)) return false;
				} catch (out_of_bounds &e) {
					// child doesn't want to accept it
					if (!m_values.insert(std::make_pair(t, a)).second) return false;
				}
			}
		}
		m_count++;
		return true;
	}

};

inline void OctreeAccel::Node::unleafify() {

}

OctreeAccel::OctreeAccel(const vector<Reference<Primitive>> &prims) {
	
	Assert(!prims.empty());

	// fully refine- what does this do?
	vector<Reference<Primitive>> allprims;
	for (const auto &prim : prims) {
		prim->FullyRefine(allprims);
	}
	
	// ???
	for (const auto &prim : allprims) {
		Assert(prim->CanIntersect());
	}

	// find bounds of whole scene
	BBox bound = prims[0]->WorldBound();
	for (const auto &prim : allprims) {
		bound = Union(bound, prim->WorldBound());
	}

	// make root
	root = new Node(aabb::fromBBox(bound));

	// add things
	for (const auto &prim : allprims) {
		root->insert(prim, aabb::fromBBox(prim->WorldBound()));
	}

	Warning("Octree built");

}

OctreeAccel::~OctreeAccel() {
	if (root) delete root;
};

BBox OctreeAccel::WorldBound() const {
	if (!root) return BBox();
	return BBox(root->bound().min(), root->bound().max());
}

bool OctreeAccel::Intersect(const Ray &ray, Intersection *isect) const {
	// TODO
	return false;
}

bool OctreeAccel::IntersectP(const Ray &ray) const {
	// TODO
	return false;
}
