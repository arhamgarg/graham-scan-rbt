#include "../include/avl.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>

int cross(Point a, Point b, Point c) {
  const __int128 value =
      (static_cast<__int128>(b.x) - a.x) * (static_cast<__int128>(c.y) - a.y) -
      (static_cast<__int128>(b.y) - a.y) * (static_cast<__int128>(c.x) - a.x);
  return (value > 0) - (value < 0);
}

namespace {

bool compare_nodes(const Node *a, const Node *b) {
  const bool a_upper = a->dy > 0 || (a->dy == 0 && a->dx >= 0);
  const bool b_upper = b->dy > 0 || (b->dy == 0 && b->dx >= 0);

  if (a_upper != b_upper)
    return a_upper > b_upper;

  const __int128 cross_prod = static_cast<__int128>(a->dx) * b->dy -
                              static_cast<__int128>(a->dy) * b->dx;
  if (cross_prod != 0)
    return cross_prod > 0;

  return a->distance2 < b->distance2;
}

int height(Node *n) {
  return n ? n->height : 0;
}

int get_balance(Node *n) {
  return n ? height(n->left) - height(n->right) : 0;
}

void update_height(Node *n) {
  if (n) {
    n->height = 1 + std::max(height(n->left), height(n->right));
  }
}

Node *rotate_right(Node *y) {
  Node *x = y->left;
  Node *T2 = x->right;

  x->right = y;
  y->left = T2;

  update_height(y);
  update_height(x);

  return x;
}

Node *rotate_left(Node *x) {
  Node *y = x->right;
  Node *T2 = y->left;

  y->left = x;
  x->right = T2;

  update_height(x);
  update_height(y);

  return y;
}

Node *balance_node(Node *node) {
  update_height(node);
  int balance = get_balance(node);

  // Left Left Case
  if (balance > 1 && get_balance(node->left) >= 0)
    return rotate_right(node);

  // Left Right Case
  if (balance > 1 && get_balance(node->left) < 0) {
    node->left = rotate_left(node->left);
    return rotate_right(node);
  }

  // Right Right Case
  if (balance < -1 && get_balance(node->right) <= 0)
    return rotate_left(node);

  // Right Left Case
  if (balance < -1 && get_balance(node->right) > 0) {
    node->right = rotate_right(node->right);
    return rotate_left(node);
  }

  return node;
}

Node *insert_helper(Node *node, Node *new_node, bool &success) {
  if (!node) {
    success = true;
    return new_node;
  }

  if (new_node->point == node->point) {
    success = false;
    return node;
  }

  if (compare_nodes(new_node, node)) {
    node->left = insert_helper(node->left, new_node, success);
  } else {
    node->right = insert_helper(node->right, new_node, success);
  }

  if (!success) return node;

  return balance_node(node);
}

Node *find_min(Node *node) {
  Node *curr = node;
  while (curr && curr->left) {
    curr = curr->left;
  }
  return curr;
}

Node *erase_helper(Node *node, Point point, long long dx, long long dy, __int128 distance2, bool &success) {
  if (!node) {
    success = false;
    return nullptr;
  }

  if (point == node->point) {
    success = true;
    if (!node->left || !node->right) {
      Node *temp = node->left ? node->left : node->right;
      delete node;
      return temp;
    } else {
      Node *temp = find_min(node->right);
      node->point = temp->point;
      node->dx = temp->dx;
      node->dy = temp->dy;
      node->distance2 = temp->distance2;
      bool dummy_success = false;
      node->right = erase_helper(node->right, temp->point, temp->dx, temp->dy, temp->distance2, dummy_success);
    }
  } else {
    const bool curr_upper = node->dy > 0 || (node->dy == 0 && node->dx >= 0);
    const bool p_upper = dy > 0 || (dy == 0 && dx >= 0);

    bool go_left = false;
    if (curr_upper != p_upper) {
      go_left = p_upper > curr_upper;
    } else {
      const __int128 cross_prod = static_cast<__int128>(dx) * node->dy -
                                  static_cast<__int128>(dy) * node->dx;
      if (cross_prod != 0) {
        go_left = cross_prod > 0;
      } else {
        go_left = distance2 < node->distance2;
      }
    }

    if (go_left) {
      node->left = erase_helper(node->left, point, dx, dy, distance2, success);
    } else {
      node->right = erase_helper(node->right, point, dx, dy, distance2, success);
    }
  }

  if (!node) return nullptr;

  return balance_node(node);
}

bool validate_avl(const Node *node, int &computed_height) {
  if (!node) {
    computed_height = 0;
    return true;
  }
  int left_height = 0, right_height = 0;
  if (!validate_avl(node->left, left_height)) return false;
  if (!validate_avl(node->right, right_height)) return false;

  if (node->height != 1 + std::max(left_height, right_height)) return false;
  if (std::abs(left_height - right_height) > 1) return false;

  computed_height = node->height;
  return true;
}

bool validate_sorted_avl(const Node *node, const Node *&last) {
  if (!node)
    return true;

  if (!validate_sorted_avl(node->left, last))
    return false;

  if (last && !compare_nodes(last, node)) {
    return false;
  }
  last = node;

  return validate_sorted_avl(node->right, last);
}

} // namespace

void DynamicHull::clear() {
  std::function<void(Node *)> delete_tree = [&](Node *node) {
    if (!node)
      return;
    delete_tree(node->left);
    delete_tree(node->right);
    delete node;
  };
  delete_tree(root_);
  root_ = nullptr;
  has_pivot_ = false;
  size_ = 0;
}

void DynamicHull::rebuild(std::vector<Point> points) {
  clear();
  if (points.empty())
    return;

  const auto pivot =
      std::min_element(points.begin(), points.end(), [](Point a, Point b) {
        return std::tie(a.y, a.x) < std::tie(b.y, b.x);
      });
  std::iter_swap(points.begin(), pivot);
  for (const auto point : points) {
    insert(point);
  }
}

DynamicHull::DynamicHull()
    : root_(nullptr), pivot_({0, 0}), has_pivot_(false), size_(0) {}

DynamicHull::~DynamicHull() {
  clear();
}

bool DynamicHull::insert(Point point) {
  if (!has_pivot_) {
    pivot_ = point;
    has_pivot_ = true;
    size_ = 1;
    return true;
  }

  if (std::tie(point.y, point.x) < std::tie(pivot_.y, pivot_.x)) {
    std::vector<Point> all_points = ordered_points();
    all_points.push_back(point);
    rebuild(std::move(all_points));
    return true;
  }

  if (point == pivot_)
    return false;

  long long dx = point.x - pivot_.x;
  long long dy = point.y - pivot_.y;

  Node *new_node = new Node(point, dx, dy);
  bool success = false;
  root_ = insert_helper(root_, new_node, success);

  if (success) {
    ++size_;
  } else {
    delete new_node;
  }
  return success;
}

std::vector<Point> DynamicHull::ordered_points() const {
  std::vector<Point> result;

  if (has_pivot_) {
    result.push_back(pivot_);
  }

  std::function<void(Node *)> inorder = [&](Node *node) {
    if (!node)
      return;
    inorder(node->left);
    result.push_back(node->point);
    inorder(node->right);
  };

  inorder(root_);
  return result;
}

bool DynamicHull::valid() const {
  if (!has_pivot_)
    return root_ == nullptr && size_ == 0;

  if (root_ == nullptr)
    return size_ == 1;

  int computed_height = 0;
  if (!validate_avl(root_, computed_height))
    return false;

  const Node *last = nullptr;
  return validate_sorted_avl(root_, last);
}

std::vector<Point> DynamicHull::hull(bool include_collinear) const {
  std::vector<Point> points = ordered_points();

  if (points.empty())
    return {};
  if (points.size() == 1)
    return points;

  std::vector<Point> result;
  for (const auto &p : points) {
    while (result.size() > 1) {
      int turn = cross(result[result.size() - 2], result[result.size() - 1], p);
      if (include_collinear) {
        if (turn < 0)
          result.pop_back();
        else
          break;
      } else {
        if (turn <= 0)
          result.pop_back();
        else
          break;
      }
    }
    result.push_back(p);
  }

  return result;
}

bool DynamicHull::erase(Point point) {
  if (point == pivot_) {
    if (size_ == 1) {
      has_pivot_ = false;
      size_ = 0;
      return true;
    } else if (root_ == nullptr) {
      return false;
    } else {
      auto points = ordered_points();
      points.erase(points.begin());
      rebuild(std::move(points));
      return true;
    }
  }

  long long dx = point.x - pivot_.x;
  long long dy = point.y - pivot_.y;
  __int128 distance2 =
      static_cast<__int128>(dx) * dx + static_cast<__int128>(dy) * dy;

  bool success = false;
  root_ = erase_helper(root_, point, dx, dy, distance2, success);
  if (success) {
    --size_;
  }
  return success;
}

std::size_t DynamicHull::size() const { return size_; }
