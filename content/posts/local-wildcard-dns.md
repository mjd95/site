---
title: "Using wildcard DNS locally"
date: 2018-09-27
---

At work, we have a number of dynamically provisioned backend services.
When they are created, an entry `(<dynamic-part>, <ip>)` is written to
our etcd cluster. The URLs are specified by a well-known scheme, so a
client knows which URL to dial - something like
`<dynamic-part>.<static-part>.com`. There is a reverse proxy running
providing a resolution service, and there is a wildcard DNS entry
`*.<static-part>.com` which points to that service. The remaining
resolution is done with an etcd lookup.

I sometimes need to mimic this flow locally, so I need a way of sending
all requests under a given domain to some service running on my own
machine.

Unfortunately, one cannot just add wildcard entries to `/etc/hosts`, so
a bit more work is needed. One way to solve this is to run a local DNS
server and have it "re-implement `/etc/hosts`, allowing wildcards". This
is actually pretty easy in Go, using the fantastic
[miekg/dns](https://github.com/miekg/dns) library for DNS. I'll now walk
through that with a simple example.

First let's create a backend HTTP server (`example/server.go`):

    package main
            
            import (
                "fmt"
                "log"
                "net/http"
            )
            
            func main() {
                http.HandleFunc("/", func (w http.ResponseWriter, r *http.Request) {
                    fmt.Fprintf("hello!\n")
                })
                log.Fatal(http.ListenAndServe(":8080", nil))
            }
            

Let's pretend that this is running at `foo.bar`, and create a client for
dialing it (`example/client.go`):

    package main
            
            import (
                "fmt"
                "io/ioutil"
                "net/http"
            )
            
            func main() {
                url := "http://foo.bar:8080"
            
                resp, err := http.Get(url)
                if err != nil {
                    panic(err)
                }
                defer resp.Body.Close()
            
                body, err := ioutil.ReadAll(resp.Body)
                if err != nil {
                    panic(err)
                }
                fmt.Println(string(body))
            }
            

In one terminal, do `go run example/server.go`. In another terminal, do
`go run example/client.go`. Oops! That didn't work. To fix it, we need
to add the following line to `/etc/hosts`:

    127.0.0.1    foo.bar
            

Do `go run example/client.go` again all is good.

Now suppose we change `example/client.go` to dial a random URL:

    package main
            
            import (
                "fmt"
                "time"
                "io/ioutil"
                "math/rand"
                "net/http"
            )
            
            func main() {
                rand.Seed(time.Now().UnixNano())
                url := fmt.Sprintf("http://foo-%d.bar", rand.Intn(100))
            
                resp, err := http.Get(url)
                if err != nil {
                    panic(err)
                }
                defer resp.Body.Close()
            
                body, err := ioutil.ReadAll(resp.Body)
                if err != nil {
                    panic(err)
                }
                fmt.Println(string(body))
            }
            

We would like this request to still be handled by `example/server.go`,
but of course `go run example/client.go` won't work right now. In fact,
we can't really know what the URL is going to be in advance, so we can't
fix this by adding a single entry `foo-<number>.bar` to `/etc/hosts`.
Unfortunately, we can't simply add `*.bar` to `/etc/hosts` either,
because wildcard entries are not supported. We *could* fix it by adding
entries for each of `foo-1.bar`, `foo-2.bar`, ..., `foo-100.bar`, but no
one wants to do that!

Instead, we will implement just enough of a DNS server to deal with
outgoing traffic under some wildcard domains we specify in a hosts file.
First we create the file `hosts` with content

    127.0.0.1    *.bar
            

Then create the DNS server `dns_server.go`:

    package main
            
            import (
                    "bufio"
                    "fmt"
                    "os"
                    "strings"
                    "github.com/miekg/dns"
            )
            
            const wildcardHostsFilename = "hosts"
            
            func main() {
                    dns.HandleFunc(".", handler)
                    server := &dns.Server{Addr: ":53", Net: "udp"}
                    if err := server.ListenAndServe(); err != nil {
                            panic(err)
                    }   
            }
            
            func handler(w dns.ResponseWriter, r *dns.Msg) {
                    names := getNamesFromMessage(r)
                    for _, name := range names {
                            if result := checkForMatchInFile(name); result != "" {
                                    m := new(dns.Msg)
                                    m.SetReply(r)
                                    rr, err := dns.NewRR(fmt.Sprintf("%s A %s", name, result))
                                    if err == nil {
                                            m.Answer = append(m.Answer, rr) 
                                    }   
                                    w.WriteMsg(m)
                            }
                    }
            
                    defaultMux := dns.NewServeMux()
                    defaultMux.ServeDNS(w, r)
            }
            
            // snip
            

This uses the external dependency `miekg/dns`, so you'll want to do
`go get github.com/miekg/dns`. Since the DNS server runs on a
privileged port, you'll need `sudo` to run it. You'll probably want to
forward your environment variables (particularly the `GOPATH`) to the
`sudo` environment, so you can run this with
`sudo -E bash -c 'go run dns_server.go`.

The only thing that remains is to actually use this DNS server, so add
an entry to the top of your `/etc/resolv.conf` with

    nameserver 127.0.0.1
            

Now with both `dns_server.go` and `example/server/server.go` running,
you can run `example/client/client.go` again and it will work, no matter
what URL is randomly generated!

See [the code](http://www.github.com/mjd95/local-wildcard-dns) on GitHub
for the full example.

</div>
