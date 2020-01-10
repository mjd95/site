---
title: "Yet Another OAuth2 Explanation"
date: 2020-01-10T13:10:00+01:00
draft: true
---

There is no shortage of explanations of OAuth2 online, but given the potentially confusing terms, I think it helps to kind of work it out for yourself.  So here is my version.

As a running example, imagine a web application running at https://contactslist.app that pretty-renders your Google contact list.

In this situation, the roles from the [OAuth2 RFC](https://tools.ietf.org/html/rfc6749#section-1.1) would be as follows:

  * _resource owner_: that would be you, the person who just pointed their browser at https://contactlist.app
  * _client_: the application, the thing you are using when you visit https://contactlist.app
  * _authorisation server_: a Google server somewhere that's capable of issuing access tokens, and will do so if it thinks its correct to do so
  * _resource server_: a Google server somewhere that's capable of return lists of contacts, and will do so in response to requests with valid access tokens

# How the client gets an access token

The _client_ wants your resources, and the resource server is going to demand an access token for that.  There are four basic flows by which the client can retrieve an access token in the OAuth2 spec.  These four flows correspond to the different types of _authorization grant_, which we now describe in turn.

## Authorization code

A standard flow for the authorization code flow is:

  * _Resource owner_ visits _client_
    - For example, you visit https://contactlistapp.business
  * The _client_ redirects the _resource owner_ to the _authorization server_.  The _client_ specifies that it using the _authorization code_ flow.  It passes along a list of _scopes_ that it is interested in, the _client ID_, and a _redirect URL_.
    - For example, you are redirected to a page hosted on some Google domain, which asks whether you want to grant the ContactListApp (which Google has determined by the _client ID_ it received) the ability to read your contact list (which Google has determined by the _scopes_ it received)
  * The _resource owner_ authorizes this access
    - For example, you type in your Google password
  * The _authorization server_ redirects the _resource owner_ to the _redirect URL_.  It passes along an _authorization code_
  * The _client_ makes another request to the _authorization server_.  This time it provides the _client ID_, the _client secret_, and the _authorization code_
  * The _authorization server_ checks the _client ID_ and the _client secret_ in order to authenticate the _client_.  It also checks the validty of the _authorization code_, and if it is still happy, it returns an _access token_
  * The _client_ makes a request to the _resource server_ with the _access token_, and gets back the _resource owner_'s _resources_ in response
    - The four previous steps are probably a sequence of callbacks initiated from the _redirect URL_ handler, and are opaque to the user (at least assuming the user does not have the network tab open).  The next thing they see is probably the pretty-rendered contact list

The _authorization code_ flow is the ideal means to obtain an access token, as it allows short-lived and restricted access (a goal of OAuth2) to the _resources_, and it also allows the identity of the _client_ to be verified (by means of the _client ID_ and the _client secret_).

## Implicit

Depending on where exactly the _client_ is running, they _authorization code_ flow might not give any additional security benefits.  If the _client_ is a bunch of backend code on the server running https://contactlist.app then it makes sense to give that _client_ an identity (the _client ID_ and _client secret_).

However, if all of the logic for https://contactlist.app is front-end Javascript code, then the _client secret_ would have to be injected into the Javascript code and would no longer be secret.  Similar reasoning applies to other client-side apps (binaries run from the command line, mobile apps, etc.).

Since there is no additional security benefit, we might as well simplify the flow as follows:

  * _Resource owner_ visits _client_
    - For example, you visit https://contactlist.app
  * The _client_ redirects the _resource owner_ to the _authentication server_.  The _client_ specifies that it is using the _implicit_ flow.  It passes along a list of _scopes_ that it is interested in, the _client ID_, and a _redirect URL_.
    - As before, you see a page on a Google domain asking if ContactListApp may access your contact list
  * The _resource owner_ authorizes this access
    - For example, you type in your Google password
  * The _authorization server_ redirects the _resource owner_ to the _redirect URL_.  It passes along an _access token_
  * The _client_ makes a request to the resource server with the _access token_, and gets back the _resource owner_'s _resources_
    - The user sees the result of the _client_ processing the _resources_ it received: a pretty-rendered contact list

## Resource Owner Password Credentials

This is what it sounds like, and should obviously be used with extreme care.  If the _client_ is in posession of the the _resource owner_'s username and password, then hopefully it has been authorized to access the _resource owner_'s resources!

The protocol allows that the _client_ can specify that it has the _resource owner_'s username and password by making a request to the _authorization server_ of type _passwordcredentials_.  The _authorization server_ can then issue an access token in response, and the _client_ uses this as in the above two examples.

## Client Credentials

There may also be a situation in which the _client_ wants to access the _resource owner_'s resources without having the _resource owner_ in the loop.  Presumably _client_ and the _authorisation server_ have a prior agreement that there will be some resources in the _resource server_'s database that will be access-controlled with access tokens.  The policy deciding whether an access token should be issued from the _authorisation server_ will presumably require the _client_ to provide some sort of credentials (for example, a _client ID_ and a _client secret_).

To handle this, the protocol allows that the _client_ can specify that it wants to access the resources using _client credentials_.  The _resource owner_ has no input whatsoever, it's purely down to the agreement between the _client_ and the _authorization server_.
