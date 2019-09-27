---
title: "A project using Rust FFI and CLI helpers"
date: 2018-11-27T07:24:00+01:00
---

I've been learning Rust recently. I started off with the [Rust
book](https://doc.rust-lang.org/book/second-edition/index.html), which
is excellent, particularly for people like me who don't have a huge
amount of experience in lower level languages. (I could well imagine
more experienced readers would find it quite verbose, though). The final
project in the book was building a small multithreaded webserver, which
I think is a really good thing to build when assessing any new language.

My only issue with the Rust book was that it does give you quite a lot
of hints. Particularly for the borrowing rules, I felt like I had some
understanding of how they worked, but knew it would be harder were I
writing all the code myself. I knew I had to be introducing my own
borrowing errors to start getting a real feel for how the language
works.

I had a decent small project ready to go. I've been getting frustrated
by having to start Python every time I want to manually hash or check a
password matches a hash with [bcrypt](https://pypi.org/project/bcrypt/).
I thought that it would be a good starting point with Rust to make a CLI
tool, exposing the same API as the Python library but consumable
directly from the terminal.

So I made [bcrust](https://github.com/mjd95/bcrust). It uses `clap` to
structure the command line part, which seems like a pretty nice way to
get argument parsing and help in place quickly. To understand how to
call the C code actually implementing `bcrypt` I used both the
[Rustonomicon](https://doc.rust-lang.org/nomicon/) and [Alex Crichton's
FFI
examples](https://github.com/alexcrichton/rust-ffi-examples/tree/master/rust-to-c),
with the latter being particularly helpful explaining how to build the C
code along as part of `cargo build`.

Most of the times I got confused during this project were somewhere
between the Rust and C interface, particularly what exactly was being
passed from the former to the latter. For example, if I invoked

    bcrust checkpw my_password <some_hash>
        

then Rust would happily extract a `&str` `my_password`. I pass then pass
this to C as a slice of bytes. In C world, you have a `*const char`,
which you think of as a string by marking the end with a null byte.
However, when C attempted to read the string in this way, it would
always interpret the password as `my_passwordcheckpw`. I'm not really
sure why this was happening, presumably something to do with how `clap`
was storing the arguments that it had parsed. In order to get around it,
I had to explicitly set a null byte at the end of what I wanted to pass
from Rust to C. There were a few segmentation faults along the way too,
but these were generally fixed by going back and reading the Rust FFI
docs.

Overall this was a nice small project, and I feel like I know a little
bit more about Rust now. There's a few more things I'd like to do with
it (clean up the code, add some tests), but it's at least at the
"working" stage now. The code is
[here](https://github.com/mjd95/bcrust).

</div>
