import json

with open("repositories.json", "r") as file:
    repositories = json.load(file)

size_range = (None, None)
watchers_range = (None, None)
forks_range = (None, None)
open_issues_range = (None, None)
contributors_range = (20, None)
contributions_range = (None, None)

filtered_repositories = list(filter(
    lambda repo: (
        (size_range[0] is None or size_range[0] <= repo["size"])
        and (size_range[1] is None or repo["size"] <= size_range[1])

        and (watchers_range[0] is None or watchers_range[0] <= repo["watchers_count"])
        and (watchers_range[1] is None or repo["watchers_count"] <= watchers_range[1])

        and (forks_range[0] is None or forks_range[0] <= repo["forks_count"])
        and (forks_range[1] is None or repo["forks_count"] <= forks_range[1])

        and (open_issues_range[0] is None or open_issues_range[0] <= repo["open_issues_count"])
        and (open_issues_range[1] is None or repo["open_issues_count"] <= open_issues_range[1])

        and (contributors_range[0] is None or contributors_range[0] <= len(repo["total_contributors"]))
        and (contributors_range[1] is None or len(repo["total_contributors"]) <= contributors_range[1])

        and (contributions_range[0] is None or contributions_range[0] <= sum([contributor["contributions"] for contributor in repo["total_contributors"]]))
        and (contributions_range[1] is None or sum([contributor["contributions"] for contributor in repo["total_contributors"]]) <= contributions_range[1])
    ),
    repositories
))

print(f"# of filtered repositories: {len(filtered_repositories)}")

with open("filtered_repositories.json", "w") as file:
    json.dump(filtered_repositories, file, indent=2)
