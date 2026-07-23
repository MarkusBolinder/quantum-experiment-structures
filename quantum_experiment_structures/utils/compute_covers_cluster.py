import itertools
import json
import sys

from quantum_experiment_structures.data.integer_sequences import NUMBER_OF_COVERS
from pyspark.sql import SparkSession


def get_all_subsets(measurements):
    """Generate all non-empty subsets."""
    subsets = []
    for r in range(1, len(measurements) + 1):
        for comb in itertools.combinations(measurements, r):
            subsets.append(frozenset(comb))
    return subsets


def dfs_extend(current_collection, current_union, start_idx, subsets, ground_set):
    """Search for antichains with DFS and pruning."""
    results = []

    # if the current collection covers the ground set, record it
    # do NOT return early (adding more independent subsets could yield more covers)
    if current_union == ground_set:
        readable = sorted([sorted(list(c)) for c in current_collection])
        results.append(readable)

    # try to extend antichain
    for i in range(start_idx, len(subsets)):
        candidate = subsets[i]

        # prune branch if candidate conflicts with any existing contexts
        if any(candidate <= c or c <= candidate for c in current_collection):
            continue

        results.extend(
            dfs_extend(
                current_collection + [candidate],
                current_union | candidate,
                i + 1,
                subsets,
                ground_set,
            )
        )

    return results


def process_seed_partition(iterator, subsets_broadcast, ground_set_broadcast):
    """Worker function processing a batch of seeds."""
    subsets = subsets_broadcast.value
    ground_set = ground_set_broadcast.value
    partition_results = []

    for seed in iterator:
        indices, next_idx = seed
        current_collection = [subsets[idx] for idx in indices]
        current_union = set().union(*current_collection)

        # run the local pruned search for this branch
        branch_covers = dfs_extend(current_collection, current_union, next_idx, subsets, ground_set)
        partition_results.extend(branch_covers)
    return partition_results


def main():
    """Initialize and run Spark job to compute all covers."""
    n = int(sys.argv[1])
    measurements = list(range(n))
    ground_set = frozenset(measurements)

    spark = SparkSession.builder.appName("compute-covers").master("local[*]").getOrCreate()

    sc = spark.sparkContext

    print(f"Starting computation for n={n} measurements...", flush=True)

    subsets = get_all_subsets(measurements)
    print(f"Total non-empty subsets to consider: {len(subsets)}", flush=True)

    # broadcast the base lookup structures to all workers safely
    subsets_bc = sc.broadcast(subsets)
    ground_set_bc = sc.broadcast(ground_set)

    # handle trivial cover in driver
    base_covers = []
    for s in subsets:
        if s == ground_set:
            base_covers.append([sorted(list(s))])

    # use all valid antichain pairs for task distribution
    seeds = []
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            if not (subsets[i] <= subsets[j] or subsets[j] <= subsets[i]):
                seeds.append(([i, j], j + 1))

    print(f"Generated {len(seeds)} independent task seeds.", flush=True)

    num_partitions = min(len(seeds), 200)
    seeds_rdd = sc.parallelize(seeds, num_partitions)

    distributed_covers_rdd = seeds_rdd.mapPartitions(
        lambda s_iter: process_seed_partition(s_iter, subsets_bc, ground_set_bc)
    )

    print("Gathering and deduplicating results on driver...", flush=True)
    base_covers_rdd = sc.parallelize(base_covers)
    # make the arrays hashable
    hashable_distributed = distributed_covers_rdd.map(lambda cover: tuple(tuple(x) for x in cover))
    hashable_base = base_covers_rdd.map(lambda cover: tuple(tuple(x) for x in cover))

    all_covers_rdd = hashable_distributed.union(hashable_base)
    unique_covers_rdd = all_covers_rdd.distinct()
    sorted_covers_rdd = unique_covers_rdd.sortBy(lambda x: x)  # type: ignore

    # make records well-formed JSON
    final_covers_rdd = sorted_covers_rdd.map(lambda cover: [list(x) for x in cover])
    final_count = final_covers_rdd.count()

    if final_count == NUMBER_OF_COVERS[n - 1]:
        print("Verification successful. Saving results...")
        if n < 7:
            all_covers_list = final_covers_rdd.collect()
            with open(f"covers_{n}.json", "w") as f:
                json.dump(all_covers_list, f, separators=(",", ":"))
            print(f"Saved to covers_{n}.json successfully.")
        else:
            final_covers_rdd.saveAsTextFile(f"covers_{n}")
        print(f"SUCCESS: Found {unique_covers_rdd.count()=} valid local covers.", flush=True)
    else:
        print(f"Count of covers does not match {NUMBER_OF_COVERS[n - 1]=}. (Got {final_count=}.)")
        print("Aborting job without saving results.")

    spark.stop()


if __name__ == "__main__":
    main()
