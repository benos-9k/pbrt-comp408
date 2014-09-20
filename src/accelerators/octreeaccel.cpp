
#include "octreeaccel.h"
#include "intersection.h"

// SSE ftw, just cause i can
#include <xmmintrin.h>

#include <cstring>
#include <iostream>
#include <unordered_map>
#include <queue>

using namespace std;

#define BEN_OCTREE_MAX_LEAF_VALUES 16

namespace {

	unsigned octree_node_count = 0;

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

		inline __m128 data() const {
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
			return float3(*this) += rhs;
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
			return float3(*this) -= rhs;
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
			return float3(*this) *= rhs;
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

		inline float min() const {
			__m128 r1 = m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_min_ss(r3, _mm_min_ss(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r;
		}

		inline float max() const {
			__m128 r1 = m_data;
			__m128 r2 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 1));
			__m128 r3 = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(0, 0, 0, 2));
			__m128 r4 = _mm_max_ss(r3, _mm_max_ss(r2, r1));
			float r;
			_mm_store_ss(&r, r4);
			return r;
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
			// intersects + contains in 1 dimension
			return float3::any(float3::abs(a.m_center - m_center) <= (m_halfsize - a.m_halfsize)) && intersects(a);
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

		inline operator BBox() const {
			return BBox(min(), max());
		}

	};

	inline void sanitize_intersection_times(float3 &mint, float3 &maxt) {
		// deal with plane-aligned rays (nans)
		// we want a pair of infinities
		// we have at least one infinity; the value for the other plane is either
		// the opposite signed infinity if the origin is between the plane pair,
		// the same signed infinity if outside the the plane pair,
		// or a nan if the origin is on one of the planes (in which case the infinity is of indeterminate signedness)
		// so - replace nans with infinity of opposite sign to t for other plane

		__m128 mint0 = mint.data();
		__m128 maxt0 = maxt.data();

		// TODO does cmpunord have the behaviour i expect?

		// replace nans in min
		__m128 minnans = _mm_cmpunord_ps(mint0, mint0);
		__m128 mint1 = _mm_or_ps(_mm_and_ps(minnans, _mm_sub_ps(_mm_setzero_ps(), maxt0)), _mm_andnot_ps(minnans, mint0));

		// replace nans in max
		__m128 maxnans = _mm_cmpunord_ps(maxt0, maxt0);
		__m128 maxt1 = _mm_or_ps(_mm_and_ps(maxnans, _mm_sub_ps(_mm_setzero_ps(), mint0)), _mm_andnot_ps(maxnans, maxt0));

		// sort entrance/exit properly
		mint = float3(_mm_min_ps(mint1, maxt1));
		maxt = float3(_mm_max_ps(mint1, maxt1));

	}

	inline bool intersection_time(const float3 &mint, const float3 &maxt, float &t0, float &t1) {
		t0 = mint.max();
		t1 = maxt.min();
		return t0 <= t1;
	}
}

class OctreeAccel::Node {
public:
	using value_t = Reference<Primitive>;

	struct hash {
		inline size_t operator()(const Reference<Primitive> &r) const {
			return size_t(r.GetPtr());
		}
	};

	struct keyeq {
		// reference doesnt have a == operator?
		inline bool operator()(const Reference<Primitive> &lhs, const Reference<Primitive> &rhs) const {
			return lhs.GetPtr() == rhs.GetPtr();
		}
	};

	using map_t = std::unordered_map<Reference<Primitive>, aabb, hash, keyeq>;

private:
	aabb m_bound;
	Node *m_children[8];
	map_t m_values;
	size_t m_count = 0;
	bool m_isleaf = true;

	inline unsigned childID(const float3 &p) const {
		return 0x7 & _mm_movemask_ps((p >= m_bound.center()).data());
	}

	// mask is true where cid bit is _not_ set
	inline __m128 childInvMask(unsigned cid) const {
		__m128i m = _mm_set1_epi32(cid);
		m = _mm_and_si128(m, _mm_set_epi32(0, 4, 2, 1));
		m = _mm_cmpeq_epi32(_mm_setzero_si128(), m);
		return _mm_castsi128_ps(m);
	}

	inline aabb childBound(unsigned cid) const {
		// positive / negative halfsizes
		__m128 h = m_bound.halfsize().data();
		__m128 g = _mm_sub_ps(_mm_setzero_ps(), h);

		// convert int bitmask to (opposite) sse mask
		__m128 n = childInvMask(cid);

		// vector to a corner of the current node's aabb
		float3 vr(_mm_or_ps(_mm_and_ps(n, g), _mm_andnot_ps(n, h)));
		const float3 c = m_bound.center();

		return aabb::fromPoints(c, c + vr);
	}
			
	inline void unleafify();

public:
	inline Node(const aabb &a_) : m_bound(a_) {
		// clear child pointers
		std::memset(m_children, 0, 8 * sizeof(Node *));
		octree_node_count++;
	}

	Node(const Node &) = delete;
	Node & operator=(const Node &) = delete;

	inline aabb bound() {
		return m_bound;
	}

	inline size_t count() {
		return m_count;
	}

	// insert a value, using a specified bounding box
	inline bool insert(const value_t &t, const aabb &a) {
		if (m_isleaf && m_count < BEN_OCTREE_MAX_LEAF_VALUES) {
			if (!m_values.insert(std::make_pair(t, a)).second) {
				Warning("Octree insert failed");
				return false;
			}
		} else {
			// not a leaf or should not be
			unleafify();
			unsigned cid_min = childID(a.min());
			unsigned cid_max = childID(a.max());
			if (cid_min != cid_max) {
				// element spans multiple child nodes
				unsigned insert_count = 0;
				unsigned cp_count = 0, isct_count = 0;

				// insert into all children that partially contain it (create duplicates)
				// create children as needed
				for (unsigned cid = 0; cid < 8; cid++) {
					aabb cb = childBound(cid);
					if (cb.intersects(a)) isct_count++;
					if (cb.contains_partial(a)) {
						cp_count++;
						Node *child = m_children[cid];
						if (!child) {
							child = new Node(cb);
							m_children[cid] = child;
						}
						insert_count += child->insert(t, a);
					}
				}
				
				// this relies on the assumption that if a value is partially contained by
				// any child, then all children that intersect it also partially contain it
				if (cp_count && (cp_count != isct_count)) Error("OctreeAccel: Assumption invalid!");

				if (!cp_count) {
					// if no child partially contains it
					// (it crosses the center-point, any face mid-point or any edge mid-point)
					// add to this node
					insert_count += m_values.insert(std::make_pair(t, a)).second;
				}

				if (!insert_count) return false;

			} else {
				// element contained in one child node - create if necessary then insert
				Node *child = m_children[cid_min];
				if (!child) {
					child = new Node(childBound(cid_min));
					m_children[cid_min] = child;
				}
				if (!child->insert(t, a)) return false;
			}
		}
		// because of allowing duplicates, counts of subtrees will no longer add up
		m_count++;
		return true;
	}

	inline bool intersect(const Ray &ray, Intersection *isct, bool fast, unsigned depth = 0) const {
		struct foo {
			float t;
			Node *n;

			inline foo(float t_, Node *n_) : t(t_), n(n_) { }

			inline bool operator<(const foo &rhs) const {
				// std::priority_queue has the _last_ element as the top
				return t > rhs.t;
			}

			inline bool intersect(const Ray &ray, Intersection *isct, bool fast, unsigned depth) const {
				return n->intersect(ray, isct, fast, depth);
			}
		};

		bool hit = false;

		// test primitives in this node
		for (auto pair : m_values) {
			// IMPORTANT: ray.maxt gets set by Primitive::Intersect()
			hit |= pair.first->Intersect(ray, isct);
			if (hit && fast) return true;
		}

		// prepare a queue of children to test in order
		std::priority_queue<foo> q;

		float3 rayo(ray.o);
		float3 irayd = float3(1.f) / float3(ray.d);

		// you can, in theory, calculate intersection times for the root
		// and then successively average them (for each axis) for child nodes
		// i havent been able to get this to work, and it has problems regarding axis-aligned rays

		for (unsigned cid = 0; cid < 8; cid++) {
			Node *child = m_children[cid];
			if (!child) continue;

			// min/max for child bounding box
			float3 minv1 = child->bound().min();
			float3 maxv1 = child->bound().max();

			// individual plane intersection times
			float3 mint1 = (minv1 - rayo) * irayd;
			float3 maxt1 = (maxv1 - rayo) * irayd;

			// sanitize in case of nans
			sanitize_intersection_times(mint1, maxt1);

			// find intersection times
			float t0, t1;
			if (intersection_time(mint1, maxt1, t0, t1)) {
				// we hit the child node
				if (t0 < ray.maxt && t1 >= ray.mint) {
					// we enter the child before ray-max (== current earliest intersection)
					// and leave the child after ray-min
					q.push(foo(t0, child));
				}
			}

			// for debugging
			//if (BBox(child->bound()).IntersectP(ray, &t0, &t1)) {
			//	q.push(foo(t0, child, mint1, maxt1));
			//}

		}

		// now test child nodes
		while (!q.empty()) {
			foo bar = q.top();
			q.pop();
			if (bar.intersect(ray, isct, fast, depth + 1)) {
				return true;
			}
		}

		// THIS IS SLIGHTLY BUGGY

		return hit;
	}

};

inline void OctreeAccel::Node::unleafify() {
	if (m_isleaf) {
		//Warning("unleafify");
		m_isleaf = false;
		// move current elements out of node
		map_t temp = std::move(m_values);
		// decrement the count because re-inserting will increment it again
		m_count -= temp.size();
		// re-insert values
		for (auto it = temp.begin(); it != temp.end(); ++it) {
			insert(std::move(it->first), it->second);
		}
	}
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

	Warning("Octree building...");

	// add things
	for (const auto &prim : allprims) {
		root->insert(prim, aabb::fromBBox(prim->WorldBound()));
	}

	Warning("Octree built (%d nodes)", octree_node_count);

}

OctreeAccel::~OctreeAccel() {
	if (root) delete root;
}

BBox OctreeAccel::WorldBound() const {
	if (!root) return BBox();
	return BBox(root->bound().min(), root->bound().max());
}

bool OctreeAccel::Intersect(const Ray &ray, Intersection *isect) const {
	if (!root) return false;

	return root->intersect(ray, isect, false);
}

bool OctreeAccel::IntersectP(const Ray &ray) const {
	if (!root) return false;
	Intersection isct;
	return root->intersect(ray, &isct, true);
}
