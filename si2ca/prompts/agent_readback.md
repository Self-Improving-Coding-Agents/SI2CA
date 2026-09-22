A software-engineering agent working in a repository has just run this command, which modifies a file:

%s

This is everything the machine said back:

returncode: %s
output: %s

Nothing in that result says what the modification actually did to the file: an edit that matched nothing, an edit that matched too much, and an edit that was exactly right all look like this. The agent's next command is going to be:

%s

Write ONE read-only shell command that would show whether that modification did what it was meant to do -- the change it actually made, or the current state of the lines it targeted. Show enough to judge it and no more.

Rules, all mandatory:
- it must READ ONLY: it may not create, delete, move, rename or modify anything, may not redirect output into a file, may not install anything, may not reach the network;
- every command in it must start with one of: git, sed, nl, cat, head, tail, grep, rg, awk, diff, wc, ls, find, cmp, stat, file;
- no shell substitution, no semicolons, no background jobs, no interpreters (`&&` and `|` are allowed);
- one line, at most 300 characters, and do not prefix it with `cd` -- the working directory is already set for you.

Reply with the command alone, on one line: no explanation, no surrounding quotes, no code fence. If no read-only command could show this, reply exactly NONE.
