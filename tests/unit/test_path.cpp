#include "gtest.h"

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <util/path.hpp>

using namespace nest::mc::util;

TEST(path, posix_ctor) {
    // test constructor ans assignment overloads with sample character sequences.
    posix::path p1;
    EXPECT_EQ("", p1.native());
    EXPECT_TRUE(p1.empty());

    const char* cs = "foo/bar";
    std::string str_cs(cs);
    std::vector<char> vec_cs(str_cs.begin(), str_cs.end());

    posix::path p2(cs);
    posix::path p3(str_cs);
    posix::path p4(str_cs.begin(), str_cs.end());
    posix::path p5(vec_cs.begin(), vec_cs.end());

    EXPECT_FALSE(p2.empty());
    EXPECT_EQ(str_cs, p2.native());
    EXPECT_EQ(str_cs, p3.native());
    EXPECT_EQ(str_cs, p4.native());
    EXPECT_EQ(str_cs, p5.native());

    posix::path p6(p2);
    EXPECT_EQ(str_cs, p6.native());

    posix::path p7(std::move(p6));
    EXPECT_EQ(str_cs, p7.native());

    // test operator= overloads (and ref return values)
    posix::path p;
    EXPECT_EQ(str_cs, (p=p2).native());
    EXPECT_EQ(str_cs, (p=cs).native());
    EXPECT_EQ(str_cs, (p=str_cs).native());
    EXPECT_EQ(str_cs, (p=vec_cs).native());
    EXPECT_EQ(str_cs, (p=std::move(p7)).native());

    // test assign overloads (and ref return values)
    EXPECT_EQ(str_cs, p.assign(p2).native());
    EXPECT_EQ(str_cs, p.assign(cs).native());
    EXPECT_EQ(str_cs, p.assign(str_cs).native());
    EXPECT_EQ(str_cs, p.assign(vec_cs).native());
    EXPECT_EQ(str_cs, p.assign(vec_cs.begin(), vec_cs.end()).native());
}

TEST(path, posix_native) {
    // native path should match string argument exactly
    std::string ps = "/abs/path";
    std::string qs = "rel/path.ext";

    EXPECT_EQ(ps, posix::path{ps}.native());
    EXPECT_EQ(qs, posix::path{qs}.native());

    // default string conversion
    std::string ps_bis = posix::path{ps};
    std::string qs_bis = posix::path{qs};

    EXPECT_EQ(ps, ps_bis);
    EXPECT_EQ(qs, qs_bis);

    // cstr
    const char *c = posix::path{ps}.c_str();
    EXPECT_TRUE(!std::strcmp(c, ps.c_str()));
}

TEST(path, posix_generic) {
    // expect native and generic to be same for POSIX paths
    path p("/abs/path"), q("rel/path.ext");

    EXPECT_EQ(p.generic_string(), p.native());
    EXPECT_EQ(q.generic_string(), q.native());
}

TEST(path, posix_append) {
    posix::path p1{""}, q1{"rel"};

    posix::path p(p1);
    p.append(q1);
    EXPECT_EQ(p1/q1, p);

    p = p1;
    p /= q1;
    EXPECT_EQ(p1/q1, p);
    EXPECT_EQ("rel", p.native());

    posix::path p2{"ab"}, q2{"rel"};

    p = p2;
    p.append(q2);
    EXPECT_EQ(p2/q2, p);

    p = p2;
    p /= q2;
    EXPECT_EQ(p2/q2, p);
    EXPECT_EQ("ab/rel", p.native());

    EXPECT_EQ("foo/bar", (posix::path("foo/")/posix::path("/bar")).native());
    EXPECT_EQ("foo/bar", (posix::path("foo")/posix::path("/bar")).native());
    EXPECT_EQ("foo/bar", (posix::path("foo/")/posix::path("bar")).native());
    EXPECT_EQ("/foo/bar/", (posix::path("/foo/")/posix::path("/bar/")).native());
}

TEST(path, compare) {
    posix::path p1("/a/b"), p2("/a//b"), p3("/a/b/c/."), p4("/a/b/c/"), p5("a/bb/c"), p6("a/b/c/");

    EXPECT_EQ(p1, p2);
    EXPECT_LE(p1, p2);
    EXPECT_GE(p1, p2);
    EXPECT_EQ(p3, p4);
    EXPECT_LE(p3, p4);
    EXPECT_GE(p3, p4);

    EXPECT_LT(p1, p3);
    EXPECT_LE(p1, p3);
    EXPECT_GT(p3, p1);
    EXPECT_GE(p3, p1);
    EXPECT_NE(p3, p1);

    EXPECT_NE(p4, p6);

    EXPECT_LT(p4, p5);
    EXPECT_LE(p4, p5);
    EXPECT_GT(p5, p4);
    EXPECT_GE(p5, p4);
    EXPECT_NE(p5, p4);
}

TEST(path, posix_concat) {
    posix::path p1{""}, q1{"tail"};

    posix::path p(p1);
    p.concat(q1);
    EXPECT_EQ("tail", p.native());

    p = p1;
    p += q1;
    EXPECT_EQ("tail", p.native());

    posix::path p2{"ab"}, q2{"cd"};

    p = p2;
    p.concat(q2);
    EXPECT_EQ("abcd", p.native());

    p = p2;
    p += q2;
    EXPECT_EQ("abcd", p.native());
}

TEST(path, posix_absrel_query) {
    posix::path p1("/abc/def");
    EXPECT_FALSE(p1.is_relative());
    EXPECT_TRUE(p1.is_absolute());

    posix::path p2("abc/def");
    EXPECT_TRUE(p2.is_relative());
    EXPECT_FALSE(p2.is_absolute());

    posix::path p3("");
    EXPECT_TRUE(p3.is_relative());
    EXPECT_FALSE(p3.is_absolute());

    posix::path p4("/");
    EXPECT_FALSE(p4.is_relative());
    EXPECT_TRUE(p4.is_absolute());

    posix::path p5("..");
    EXPECT_TRUE(p3.is_relative());
    EXPECT_FALSE(p3.is_absolute());
}

TEST(path, posix_swap) {
    posix::path p1("foo"), p2("/bar");
    p1.swap(p2);

    EXPECT_EQ("foo", p2.native());
    EXPECT_EQ("/bar", p1.native());

    swap(p1, p2);

    EXPECT_EQ("foo", p1.native());
    EXPECT_EQ("/bar", p2.native());
}

TEST(path, posix_iostream) {
    std::istringstream ss("/quux/xyzzy");
    posix::path p;
    ss >> p;
    EXPECT_EQ("/quux/xyzzy", p.native());

    std::ostringstream uu;
    uu << p;
    EXPECT_EQ("/quux/xyzzy", uu.str());
}

