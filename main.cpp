//
//  main.cpp
//  recommendation_system
//
//  Created by Yu Huang on 04/15/16.
//  Copyright (c) 2016 Yu Huang. All rights reserved.
//

#include <time.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>
/* --uncomment to gain multithreading feature--
#include <omp.h>*/

using namespace std;

struct item {
    int Id;
    // can be ommited here
    double avg_rating;
    long int salesrank;
    // store the list of co-purchasing items
    vector<string> similars;
    // store reviews of the current item instance
    vector<string> user_review_id;
    item() {
        Id = 0;
        salesrank = 0;
    }
};

struct user {
    string ID;
    // store reviews made by the current user instance
    vector<int> review_item_id;
    // store the rating of purchased items, corresponding by vector index
    vector<int> review_item_rating;
    user() {
        ID = "";
    }
};

// split a line by space, returning token
vector<string> split(const string& line) {
    vector<string> tokens;
    stringstream stream(line);
    string cur;
    while (getline(stream, cur, ' ')) {
        // skip title and category detailed
        if (cur.length() > 0 && (cur == "categories:" || cur == "group:" || cur == "title:" || cur.at(0) == '|')) {
            break;
        }
        if (cur.length() != 0) {
            tokens.push_back(cur);
        }
    }
    return tokens;
}

item by_salesrank(vector<item>& items) {
    size_t n = items.size();
    item highest;
    if (n == 0) {
        return highest;
    } else {
        highest = items[0];
        for (int i = 1; i < n; i++) {
            if (highest.salesrank < items[i].salesrank) {
                highest = items[i];
            }
        }
    }
    return highest;
}

// compute the pearson correlation between two users
double compute_user_pearson(user& selected_user, user& comp_user) {
    // store common reviewed item, matched by index
    vector<int> s_ratings;
    vector<int> comp_ratings;
    // find common reviewed product
    for (int i = 0; i < selected_user.review_item_id.size(); i++) {
        int s_item_id = selected_user.review_item_id[i];
        for (int j = 0; j < comp_user.review_item_id.size(); j++) {
            int comp_item_id = comp_user.review_item_id[j];
            if (s_item_id == comp_item_id) {
                s_ratings.push_back(selected_user.review_item_rating[i]);
                comp_ratings.push_back(comp_user.review_item_rating[j]);
                break;
            }
        }
    }
    if (s_ratings.size() == 0) {
        return 1;
    }
    double sum1 = 0;
    double sum2 = 0;
    double sum1_sq = 0;
    double sum2_sq = 0;
    double multi_sum = 0;
    for (int i = 0; i < s_ratings.size(); i++) {
        sum1 += s_ratings[i];
        sum2 += comp_ratings[i];
        sum1_sq += s_ratings[i] * s_ratings[i];
        sum2_sq += comp_ratings[i] * comp_ratings[i];
        multi_sum += s_ratings[i] * comp_ratings[i];
    }
    double num = multi_sum - (sum1 * sum2 / (double)s_ratings.size());
    double density = sqrt((sum1_sq - sum1 * sum1 / s_ratings.size()) * (sum2_sq - sum2 * sum2 / s_ratings.size()));
    if (density == 0) {
        return 0.0;
    }
    return num / density;
}

// compute pearsons among users with highest similarities
void compute_pearsons(user& selected_user, vector<user>& users, vector<user>& pearson_ids, vector<double>& top_pearsons) {
    for (int i = 0; i < users.size(); i++) {
        user comp_user = users[i];
        if (comp_user.ID != selected_user.ID) {
            double pearson = compute_user_pearson(selected_user, comp_user);
            if (pearson > 0.3) {
                if (top_pearsons.size() == 0 || pearson > top_pearsons[0]) {
                    top_pearsons.insert(top_pearsons.begin(), pearson);
                    pearson_ids.insert(pearson_ids.begin(), comp_user);
                } else if (pearson > top_pearsons[top_pearsons.size() - 1]) {
                    for (int m = 0; m < top_pearsons.size(); m++) {
                        if (pearson >= top_pearsons[m]) {
                            top_pearsons.insert(top_pearsons.begin() + m , pearson);
                            pearson_ids.insert(pearson_ids.begin() + m, comp_user);
                            break;
                        }
                    }
                }
                // delete the lowest pearson correlation when size exceeds 5
                if (top_pearsons.size() > 5) {
                    top_pearsons.pop_back();
                    pearson_ids.pop_back();
                }
            }
        }
    }
    return;
}

// calculate a list of top 10 similarities users given a user
void compute_user_similarities(user& selected_user, vector<user>& users, vector<user>& top_users, vector<double>& top_weights) {
    /* -- uncomment to reach parallel effect --
    int t = (users.size() >= 10000)? 1000: 500;
    omp_set_num_threads(t);
    #pragma omp parallel for schedule(dynamic) private(weight, comp_user)
     */
    for (int i = 0; i < users.size(); i++) {
        double weight = 0;
        user comp_user = users[i];
        if (comp_user.ID != selected_user.ID) {
            // skip users whose review item is too few
            if (comp_user.review_item_id.size() >= 2) {
                // find common review items
                for (int j = 0; j < selected_user.review_item_id.size(); j++) {
                    for (int k = 0; k < comp_user.review_item_id.size(); k++) {
                        if (selected_user.review_item_id[j] == comp_user.review_item_id[k]) {
                            int s_rating = selected_user.review_item_rating[j];
                            int comp_rating = comp_user.review_item_rating[k];
                            int dist = s_rating - comp_rating;
                            switch (abs(dist)) {
                                case 0:
                                    weight +=2;
                                    break;
                                case 1:
                                    if ((s_rating == 2 && dist == -1) || (s_rating == 3 && dist == 1)) {
                                        weight += 1;
                                    } else {
                                        weight += 1.5;
                                    }
                                    break;
                                case 2:
                                    weight -= 1;
                                    break;
                                case 3:
                                    weight -= 1.5;
                                    break;
                                case 4:
                                    weight -= 2;
                                    break;
                                default:
                                    break;
                            }
                            break;
                        }
                    }
                }
            }
        }
        // insert the weight result
        if (weight > 0) {
            if (top_weights.size() == 0 || weight > top_weights[0]) {
                top_weights.insert(top_weights.begin(), weight);
                top_users.insert(top_users.begin(), users[i]);
            } else if (weight > top_weights[top_weights.size() - 1]) {
                for (int m = 0; m < top_weights.size(); m++) {
                    if (weight >= top_weights[m]) {
                        top_weights.insert(top_weights.begin() + m , weight);
                        top_users.insert(top_users.begin() + m, users[i]);
                        break;
                    }
                }
            }
            // delete the last one from top user list and weight list
            if (top_weights.size() > 20) {
                top_weights.pop_back();
                top_users.pop_back();
            }
        }
        weight = 0.0;
    }
    return;
}

// check if the current item is already reviewed for a given user
bool item_exist(item comp, user& selected_user) {
    for (int i = 0; i < selected_user.review_item_id.size(); i++) {
        if (comp.Id == selected_user.review_item_id[i]) {
            return true;
        }
    }
    return false;
}

// return the corresponding item given an item id
item id_to_item(int id, vector<item>& items) {
    item result;
    int index = (id < (int)items.size())? id: (int)items.size() - 1;
    while (index >= 0) {
        if (items[index].Id == id) {
            return items[index];
        }
        index--;
    }
    return result;
}

// BFS to get the shortest distance
int compute_shortest_distance(user& selected_user, item& unbought, vector<item>& items, unordered_map<string, int>& asin_to_id) {
    vector<item> queue;
    queue.push_back(unbought);
    int count = 1;
    int level = 0;
    while (!queue.empty()) {
        int next_level_count = 0;
        for (int i = 0; i < count; i++) {
            item cur_item = queue.back();
            queue.pop_back();
            /* soly for debugging purpose
             item unbought_detailed = unbought;
             user show_selected_detailed = selected_user;*/
            for (int j = 0; j < cur_item.similars.size(); j++) {
                // make sure the item is in the item list because of dataset limitations
                if (asin_to_id.find(cur_item.similars[j]) != asin_to_id.end()) {
                    int convert_id = asin_to_id.find(cur_item.similars[j])->second;
                    item convert_item = id_to_item(convert_id, items);
                    // check if the neighbor is in the selected user's review list
                    for (int k = 0; k < selected_user.review_item_id.size(); k++) {
                        if (convert_id == selected_user.review_item_id[k]) {
                            return level + 1;
                        }
                    }
                    queue.push_back(convert_item);
                    next_level_count++;
                }
            }
            count = next_level_count;
        }
        level++;
    }
    // unreachable product
    level = INT_MAX;
    return level;
}

// compute the top scores
void compute_scores(user& selected_user, vector<user>& top_users, vector<double>& top_weights, vector<item>& items, unordered_map<string, int>& asin_to_id, vector<double>& top_scores, vector<int>& top_score_ids) {
    for (int i = 0; i < top_users.size(); i++) {
        user comp_user = top_users[i];
        if (comp_user.ID != selected_user.ID) {
            // find items where selected user didn't buy but top users bought
            for (int j = 0; j < selected_user.review_item_id.size(); j++) {
                int s_item_id = selected_user.review_item_id[j];
                bool comp_bought_flag = false;
                for (int k = 0; k < comp_user.review_item_id.size(); k++) {
                    int comp_item_id = comp_user.review_item_id[k];
                    if (comp_item_id == s_item_id) {
                        comp_bought_flag = true;
                    }
                }
                if (comp_bought_flag == false) {
                    item unbought = id_to_item(j, items);
                    // calculate shortest distance
                    int dist = compute_shortest_distance(selected_user, unbought, items, asin_to_id);
                    /* just for bebugging:
                    double tmp_weight = top_weights[i];*/
                    double cur_score = top_weights[i] / dist;
                    // insert to top_scores and top_score_ids lists
                    if (cur_score > 0) {
                        if (top_scores.size() == 0 || cur_score > top_scores[0]) {
                            top_scores.insert(top_scores.begin(), cur_score);
                            top_score_ids.insert(top_score_ids.begin(), j);
                        } else if (cur_score >= top_scores[top_scores.size() - 1]) {
                            for (int m = 0; m < top_scores.size(); m++) {
                                if (cur_score >= top_scores[m]) {
                                    top_scores.insert(top_scores.begin() + m, cur_score);
                                    top_score_ids.insert(top_score_ids.begin() + m, j);
                                    break;
                                }
                            }
                        }
                        // delete the one with lowest score if size exceeds five
                        if (top_score_ids.size() > 5) {
                            top_score_ids.pop_back();
                            top_scores.pop_back();
                        }
                    }
                }
            }
        }
    }
    return;
}

// extract user graph, item graph and a item ASIN to item Id map from given file
void data_parsing(const char * argv1, vector<user>& users, vector<item>& items, unordered_map<string, int>& asin_to_id) {
    string line;
    ifstream infile(argv1);
    user cur_user;
    item cur_item;
    int review_count = 0;
    while (getline(infile, line)) {
        vector<string> tokens = split(line);
        if (tokens.size() != 0) {
            if (tokens[0] == "Id:") {
                int item_id = stoi(tokens[1]);
                cur_item.Id = item_id;
                // clear last user record
                cur_user.review_item_rating.clear();
                cur_user.review_item_id.clear();
                cur_user.review_item_id.push_back(item_id);
            } else if (tokens[0] == "ASIN:") {
                asin_to_id[tokens[1]] = cur_item.Id;
            } else if (tokens[0] == "salesrank:") {
                cur_item.salesrank = stoi(tokens[1]);
            } else if (tokens[0] == "similar:") {
                int similars = stoi(tokens[1]) + 2;
                for (int i = 2; i < similars; i++) {
                    cur_item.similars.push_back(tokens[i]);
                }
            } else if (tokens[0] == "reviews:") {
                review_count = stoi(tokens[2]);
                if (review_count == 0) {
                    items.push_back(cur_item);
                }
                cur_item.avg_rating = stod(tokens[7]);
            } else {
                cur_item.user_review_id.push_back(tokens[2]);
                cur_user.ID = tokens[2];
                bool insert_flag = false;
                // check if user already exists in the user list
                for (int i = 0; i < users.size(); i++) {
                    user comp = users[i];
                    if (users[i].ID == cur_user.ID) {
                        users[i].review_item_id.push_back(cur_item.Id);
                        users[i].review_item_rating.push_back(stoi(tokens[4]));
                        insert_flag = true;
                        break;
                    }
                }
                if (insert_flag == false) {
                    cur_user.review_item_rating.push_back(stoi(tokens[4]));
                    users.push_back(cur_user);
                    cur_user.review_item_rating.clear();
                }
                review_count--;
                if (review_count == 0) {
                    items.push_back(cur_item);
                    // clear last item record
                    cur_item.similars.clear();
                    cur_item.user_review_id.clear();
                }
            }
        }
    }
    return;
}

// get the recommendation list from all top users, extracting the most common items
vector<int> get_mutual_items (vector<int> list1, vector<int> list2) {
    vector<int> mutual_item_list;
    if (list1.size() == 0 || list2.size() == 0) {
        return vector<int>();
    }
    for (int i = 0; i < list1.size(); i++) {
        for (int j = 0; j < list2.size(); j++) {
            if (list1[i] == list2[j]) {
                mutual_item_list.push_back(list1[i]);
                break;
            }
        }
    }
    return mutual_item_list;
}

// find the highest sales rank item in a user's review item
int high_sale_item (user cur_user, vector<item>& items) {
    size_t n = cur_user.review_item_id.size();
    if (n == 0) {
        return 0;
    }
    int item_id = cur_user.review_item_id[0];
    long int highest_salesrank = id_to_item(item_id, items).salesrank;
    for (int i = 1; i < n; i++) {
        long int comp_salesrank = id_to_item(cur_user.review_item_id[i], items).salesrank;
        if (comp_salesrank > highest_salesrank) {
            item_id = cur_user.review_item_id[i];
            highest_salesrank = comp_salesrank;
        }
    }
    return item_id;
}

void test_pearsons(vector<user>& selected_users, vector<user>& users, vector<item>& items, unordered_map<string, int>& asin_to_id) {
    // extract the first
    vector<int> highest_items;
}

// return a list of items where similar user bought but selected user didn't buy
vector<int> unbought_items(user& selected_user, user comp_user) {
    vector<int> unbought_item_list;
    for (int i = 0; i < comp_user.review_item_id.size(); i++) {
        int comp_item_id = comp_user.review_item_id[i];
        bool unbought_flag = true;
        for (int j = 0; j < selected_user.review_item_id.size(); j++) {
            if (comp_item_id == selected_user.review_item_id[j]) {
                unbought_flag = false;
                break;
            }
        }
        if (unbought_flag) {
            unbought_item_list.push_back(comp_item_id);
        }
    }
    return unbought_item_list;
}

// return the similarities of selected users, skipping user who only review on one items
void test_similarities(vector<user>& selected_users, vector<user>& users, vector<item>& items, unordered_map<string, int>& asin_to_id) {
    int correct_count = 0;
    int total_count = (int)selected_users.size();
    for (int i = 0; i < selected_users.size(); i++) {
        user test_user = selected_users[i];
        // remove one item with the rating the user already bought and reviewed
        /* -- uncomment to remove a random item from list, the default remove item is the last one--
         srand (time(NULL));
         int remove_index = rand() % selected_user.review_item_id.size();*/
        size_t last_item_index = test_user.review_item_id.size() - 1;
        int remove_item_id = test_user.review_item_id[last_item_index];
        int remove_item_rating = test_user.review_item_rating[last_item_index];
        test_user.review_item_id.pop_back();
        test_user.review_item_rating.pop_back();
        // simulate the similarity calculation process
        vector<user> top_users;
        vector<double> top_weights;
        compute_user_similarities(test_user, users, top_users, top_weights);
        // in list testing using similarity as a single factor, finding not buying list
        
        // adding the item back for following test
        test_user.review_item_id.push_back(remove_item_id);
        test_user.review_item_rating.push_back(remove_item_rating);
    }
    double correct_rate = (double)correct_count / (double) total_count;
    cout << "total performing tests:" << total_count << endl;
    cout << "correct rate: " << correct_rate << "%" << endl;
    return;
}

// the hybride model combining the weighted user algorithm with shortest distance
void test_hybrid_model(vector<user>& selected_users, vector<user>& users, vector<item>& items, unordered_map<string, int>& asin_to_id) {
    int correct_count = 0;
    int total_count = (int)selected_users.size();
    for (int i = 0; i < total_count; i++) {
        user test_user = selected_users[i];
        // remove one item with the rating the user already bought and reviewed
        /* -- uncomment to remove a random item from list, the default remove item is the last one--
         srand (time(NULL));
         int remove_index = rand() % selected_user.review_item_id.size();*/
        size_t last_item_index = test_user.review_item_id.size() - 1;
        int remove_item_id = test_user.review_item_id[last_item_index];
        int remove_item_rating = test_user.review_item_rating[last_item_index];
        test_user.review_item_id.pop_back();
        test_user.review_item_rating.pop_back();
        // simulate the similarity calculation process
        vector<user> top_users;
        vector<double> top_weights;
        vector<double> top_scores;
        vector<int> top_score_ids;
        // calculate the similarities
        compute_user_similarities(test_user, users, top_users, top_weights);
        // calculate the shortest distance among co-purchasing items
        compute_scores(test_user, top_users, top_weights, items, asin_to_id, top_scores, top_score_ids);
        for (int i = 0; i < top_score_ids.size(); i++) {
            if (top_score_ids[i] == remove_item_id) {
                correct_count++;
            }
        }
        // adding the item back for following test
        test_user.review_item_id.push_back(remove_item_id);
        test_user.review_item_rating.push_back(remove_item_rating);
    }
    double correct_rate = (double)correct_count / (double) total_count;
    cout << "total performing tests:" << total_count << endl;
    cout << "correct rate: " << correct_rate << "%" << endl;
    return;
}

// combining person with shortest distance
void test_hybrid_pearson_correlation(vector<user>& selected_users, vector<user>& users, vector<item>& items, unordered_map<string, int>& asin_to_id) {
    int correct_count = 0;
    int total_count = (int)selected_users.size();
    for (int i = 0; i < selected_users.size(); i++) {
        user test_user = selected_users[i];
        // remove one item with the rating the user already bought and reviewed
        /* -- uncomment to remove a random item from list, the default remove item is the last one--
         srand (time(NULL));
         int remove_index = rand() % selected_user.review_item_id.size();*/
        size_t last_item_index = test_user.review_item_id.size() - 1;
        int remove_item_id = test_user.review_item_id[last_item_index];
        int remove_item_rating = test_user.review_item_rating[last_item_index];
        test_user.review_item_id.pop_back();
        test_user.review_item_rating.pop_back();
        // simulate the similarity calculation process
        vector<double> top_pearsons;
        vector<user> pearson_ids;
        vector<double> top_scores;
        vector<int> top_score_ids;
        compute_pearsons(test_user, users, pearson_ids, top_pearsons);
        // check if the remove item is in the recommendation list
        for (int i = 0; i < top_score_ids.size(); i++) {
            if (top_score_ids[i] == remove_item_id) {
                correct_count++;
            }
        }
        // adding the item back for following test
        test_user.review_item_id.push_back(remove_item_id);
        test_user.review_item_rating.push_back(remove_item_rating);
    }
    double correct_rate = (double)correct_count / (double) total_count;
    cout << "total performing tests:" << total_count << endl;
    cout << "correct rate: " << correct_rate * 100 << "%" << endl;
    return;
}


// select a series of user whose number of reviews exceeds 5
vector<user> select_user(vector<user>& users) {
    vector<user> user_review_counts;
    for (int i = 0; i < users.size(); i++) {
        if (users[i].review_item_id.size() >= 3) {
            user_review_counts.push_back(users[i]);
        }
    }
    return user_review_counts;
}

// entrance of the program
int main(int argc, const char * argv[]) {
    vector<user> users;
    vector<item> items;
    unordered_map<string, int> asin_to_id;
    data_parsing(argv[1], users, items, asin_to_id);
    vector<user> selected = select_user(users);
    // for single testing for debuggin functions
    vector<user> top_users;
    vector<double> top_weights;
    vector<double> top_scores;
    vector<int> top_score_ids;
    vector<double> top_pearsons;
    vector<user> pearson_ids;
    compute_user_similarities(selected[5], users, top_users, top_weights);
    compute_pearsons(selected[5], users, pearson_ids, top_pearsons);
    compute_scores(selected[5], top_users, top_weights, items, asin_to_id, top_scores, top_score_ids);
    // print for bebugging
    cout << "single test ends" << endl;
    // for multiple test
    test_hybrid_model(selected, users, items, asin_to_id);
    cout<< "whole test ends" << endl;
    return 0;
}
